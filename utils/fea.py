import numpy as np
from dolfinx import fem, io, plot
import dolfinx.fem.petsc
import sympy as sp
from .symbolic import sym_legendre_series
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

from ufl import sym, grad, Identity, tr, inner, Measure, TestFunction, TrialFunction
import ufl


def import_mesh(f_name):
    '''
    Load xdmf mesh into dolfinx
    '''
    
    with XDMFFile(MPI.COMM_WORLD, f_name, 'r') as xdmf:
        mesh = xdmf.read_mesh(name='mesh')
    
    return mesh

def eval_points(domain,func,points):
    '''
    evaluate functions at points
    '''
    bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
    potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, potential_colliding_cells, points)

    points_on_proc = []
    cells = []
    # make sure to get only one point
    for ii, point in enumerate(points):
        if len(colliding_cells.links(ii)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(ii)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64).reshape(-1, 3)
    cells = np.array(cells, dtype=np.int32)
    return func.eval(points_on_proc, cells)

def apply_load(coeff, mag, a_lim_num, t):
    '''
    Interpolate legendre loads (in sympy) into FE space
    '''
    # Symypy synbols
    s = sp.symbols('s')
    m = sp.symbols(' m')
    a_lim = sp.symbols('a_lim', positive=True)
    c = sp.symbols(f'c_1:{len(coeff)+1}')
    p = sym_legendre_series(len(coeff))# applied load in sympy form

    p = p.subs({a_lim:a_lim_num, m:mag})
    p = sp.simplify(p.subs([c[ii], coeff[ii]] for ii in range(len(coeff))))

    t_num = sp.lambdify([s],p,modules=['numpy'])

    class applied_load():
        def __call__(self,x):
            return -t_num(x[0]) # negative for downward
    #---------------Define weak form----------------#
    t_applied = applied_load()
    t.sub(1).interpolate(t_applied)
    return t

def epsilon(v):
        return sym(grad(v))

def sigma(v, E, nu):
        lmbda = E*nu/(1 + nu)/(1 - 2 * nu)
        mu = E/2/(1 + nu)
        return lmbda * tr(epsilon(v)) * Identity(len(v)) + 2 * mu * epsilon(v)

def run_fea(domain, L,H, coeff):
    dim = domain.topology.dim
    degree = 2 
    V = fem.functionspace(domain, ("Lagrange", degree, (dim,))) # Function space as Lagrange shape functions with dimensions dim
    #-----------Marking Facets for Boundary conditions----------------#
    def left(x):
        return np.isclose(x[0], -L/2, 1e-6)
    def right(x):
        return np.isclose(x[0], L/2, 1e-6)
    def top(x): # only for prescribed force
        return np.isclose(x[1], H, 1e-6)
    def bot(x):
        return np.isclose(x[1], 0, 1e-6)

    # getting entities
    left_facets = dolfinx.mesh.locate_entities_boundary(domain,dim-1,left)
    right_facets = dolfinx.mesh.locate_entities_boundary(domain,dim-1,right)
    top_facets = dolfinx.mesh.locate_entities_boundary(domain,dim-1,top)
    bot_facets = dolfinx.mesh.locate_entities_boundary(domain,dim-1,bot)

    num_facets = domain.topology.index_map(dim - 1).size_local # get number of facets in system
    markers = np.zeros(num_facets, dtype=np.int32) # vector of markers -- initialize as 0
    # mark facets
    markers[left_facets] = 1 
    markers[right_facets] = 2
    markers[top_facets] = 3
    markers[bot_facets] = 4

    facet_marker = dolfinx.mesh.meshtags(domain, dim - 1, np.arange(num_facets, dtype=np.int32), markers) # mark facets

    # Mark Dirichlet BCs
    Vx, _ = V.sub(0).collapse() # x dof subspace
    Vy, _ = V.sub(1).collapse() # y dof subspace
    left_dofs = dolfinx.fem.locate_dofs_topological((V.sub(0), Vx), dim-1, facet_marker.find(1))
    right_dofs = dolfinx.fem.locate_dofs_topological((V.sub(0), Vx), dim-1, facet_marker.find(2))
    bot_dofs = dolfinx.fem.locate_dofs_topological(V, dim-1, facet_marker.find(4))

    u0 = fem.Function(V)

    bcs = [

        fem.dirichletbc(u0, bot_dofs)
    ]

    #---------------Constitutive Model------------#
    E = fem.Constant(domain, 1.0)
    nu = fem.Constant(domain, 0.0)


    #--------traction force (legendre from sympy)-----#
    #Load parameters
    mag = 1 # magnitude
    a_lim_num = L/2 # width
    t = fem.Function(V)

    t = apply_load(coeff,mag,a_lim_num,t)

    # #---------------Define weak form----------------#
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_marker) # measures

    #setup function spaces
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # Bilinear form
    a = ufl.inner(sigma(u, E,nu), epsilon(v)) * ufl.dx 
    l = ufl.dot(t, v) * ds(3) # no body force --  only traction

    #--------solve problem------------------#
    uh= fem.Function(V, name="Displacement") 
    problem = fem.petsc.LinearProblem(a, l, u=uh, bcs=bcs)
    problem.solve()
    print('Starting linear solve')
    uh.x.scatter_forward()
    print('Linear solve done!')

    return a, l, uh, bcs, u0

def compute_stress(W, u, E, nu):
    sigma_expr = fem.Expression(sigma(u, E, nu), W.element.interpolation_points())
    sigma_u = fem.Function(W)
    sigma_u.interpolate(sigma_expr)
    sigma_u.x.scatter_forward()
    return sigma_u

def compute_reaction_force(V,a,u,l,bcs,u0):
    v_reac = fem.Function(V)

    residual = ufl.action(a, u) - l

    weak_form = fem.form(ufl.action(residual,v_reac))

    def one(x):
        values = np.zeros((1,x.shape[1]))
        values[0] = 1.0
        return values
    u0.sub(1).interpolate(one)
    fem.set_bc(v_reac.x.petsc_vec,bcs)


    return fem.assemble_scalar(weak_form)





