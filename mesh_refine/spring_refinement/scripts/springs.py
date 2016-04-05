from numpy import *
from myarray import *

## This is the spring constant
k = 1.

## This is our threshold for whether a number is approximately zero.
ZERO_THRESHOLD = 1e-8

def F_ij( p_i, p_j, r_ij ):
    '''
    Returns the force of the spring from 'p_i' to 'p_j' with rest length 'r_ij' acting on 'p_i'.
    Note that F_ij( p_i, p_j, r_ij ) equals -F_ij( p_j, p_i, r_ij ).
    '''

    p_ij = p_i - p_j
    len_p_ij = sqrt( sum( p_ij ** 2 ) )

    if abs( len_p_ij ) < ZERO_THRESHOLD:
        result = 0. * p_ij
    else:
        result = -k * ( len_p_ij - r_ij ) / len_p_ij * p_ij

    return result

def F( p, edges, edges_rest_lengths ):
    '''
    Returns a vector containing the force at every point.
    Note that the input 'p' is assumed to be a number-of-points by dimension array, where
    dimension is 2 for our example.  The result is flattened into a single vector
    of size number-of-points times dimension.
    '''

    dim = p.shape[1]

    Fp = zerosf( prod( p.shape ) )

    ## Loop over every edge and its corresponding rest length.
    ## (zip() simply combines two lists into one so we can loop over them together.)
    for (i,j), r_ij in zip( edges, edges_rest_lengths ):
        assert i != j

        Fij = F_ij( p[i], p[j], r_ij )

        Fp[ i*dim : (i+1) * dim ] += Fij
        ## 'edges' contains edges uniquely, so 'edges' will contain (i,j) but not (j,i).
        ## This means that we must add -Fij to row j as well as Fij to row i.
        Fp[ j*dim : (j+1) * dim ] += -Fij

    return Fp

def dF_ij_d_p_i( p_i, p_j, r_ij ):
    '''
    Returns the derivative with respect to 'p_i' of the force of the spring
    from 'p_i' to 'p_j' with rest length 'r_ij' acting on 'p_i'.
    Our dimension is 2, so this is a 2x2 quantity.
    Note that dF_ij_d_p_i( p_i, p_j, r_ij ) equals dF_ij_d_p_i( p_j, p_i, r_ij ).
    '''

    dim = p_i.shape[0]

    p_ij = p_i - p_j
    len_p_ij = sqrt( sum( p_ij ** 2 ) )
    if abs( len_p_ij ) < ZERO_THRESHOLD:
        result = -k * identity( dim )
    else:
        result = -k * identity( dim ) - k * r_ij / len_p_ij**3 * outer( p_ij, p_ij ) + k * r_ij / len_p_ij * identity( dim )
    return result

def J( p, edges, edges_rest_lengths ):
    '''
    Returns a matrix containing the derivative of the force at every point with respect to each point.
    Note that the input 'p' is assumed to be a number-of-points by dimension array, where
    dimension is 2 for our example.
    The result is flattened is a square matrix (of type numpy.array), of size NxN, where
    N = number-of-points times dimension.
    '''

    dim = p.shape[1]

    Jp = zerosf( ( prod( p.shape ), prod( p.shape ) ) )

    ## Loop over every edge and its corresponding rest length.
    ## (zip() simply combines two lists into one so we can loop over them together.)
    for (i,j), r_ij in zip( edges, edges_rest_lengths ):
        assert i != j

        dF = dF_ij_d_p_i( p[i], p[j], r_ij )
        assert ( ( Jp[ i*dim : (i+1) * dim, j*dim : (j+1) * dim ] - zeros( ( dim, dim ) ) ) ** 2 ).sum().sum() == 0.

        Jp[ i*dim : (i+1) * dim, j*dim : (j+1) * dim ] = -dF
        Jp[ i*dim : (i+1) * dim, i*dim : (i+1) * dim ] += dF
        ## 'edges' contains edges uniquely, so 'edges' will contain (i,j) but not (j,i).
        ## This means that we must add dF to the right places in column j as well.
        Jp[ j*dim : (j+1) * dim, i*dim : (i+1) * dim ] = -dF
        Jp[ j*dim : (j+1) * dim, j*dim : (j+1) * dim ] += dF

    return Jp

def constrain_system( A, rhs, rows ):
    '''
    This function modifies its input parameters, a system matrix 'A' and
    right-hand-side vector 'rhs', such that for every index i in 'rows',
    the row i of A is set to row i of the identity matrix and rhs[i] is set to zero.
    '''

    for i in rows:
        A[ i, : ] = zeros( A.shape[1] )
        ## We can also zero the column, which keeps the matrix symmetric, because
        ## we are zeroing the corresponding entries in the right-hand-side (x*0 = 0).
        A[ :, i ] = zeros( A.shape[0] )
        A[ i, i ] = 1
        rhs[i] = 0

    return A, rhs

def static_solution( p, edges, edges_rest_lengths, constraints, verbose = True ):
    '''
    Given a list of points 'p' as an n-by-2 array, a list of (i,j) pairs 'edges' denoting an edge
    between points p[i] and p[j], a list of rest lengths (one for each edge in 'edges'),
    and a list of position constraints (i, position) denoting p[i] = position,
    uses Newton's method to solve for the positions where the forces are all zero.

    NOTE: 'edges' must not have both (i,j) and (j,i)
    '''

    XSTEP_THRESHOLD = 1e-5
    F_THRESHOLD = 1e-8
    MAX_ITERATIONS = 100

    p_n = p.copy().flatten()
    dim = p.shape[1]

    constrain_rows = []
    for i, p_val in constraints:
        p_n[ i*dim : (i+1) * dim ] = p_val
        constrain_rows.extend( range( i*dim, (i+1) * dim ) )

    iteration = 0
    while True:
        if verbose: print '-- iteration', iteration, '--'
        iteration += 1

        Jp_n = J( p_n.reshape( p.shape ), edges, edges_rest_lengths )
        Fp_n = F( p_n.reshape( p.shape ), edges, edges_rest_lengths )
        mag2_Fp_n = sum( Fp_n ** 2 )
        if verbose: print '| F( p_n ) |^2:', mag2_Fp_n
        if mag2_Fp_n < F_THRESHOLD: break

        constrain_system( Jp_n, Fp_n, constrain_rows )

        # p_n_p_1 = p_n - dot( linalg.inv( Jp_n ), Fp_n )
        ## <=> p_n_p_1 - p_n = -linalg.inv( Jp_n ) * Fp_n
        ## <=> p_n - p_n_p_1 = linalg.inv( Jp_n ) * Fp_n
        ## <=> Jp_n * ( p_n - p_n_p_1 ) = Fp_n
        p_negative_delta = linalg.solve( Jp_n, Fp_n )
        ## p_n - ( p_n - p_n_p_1 ) = p_n_p_1
        p_n_p_1 = p_n - p_negative_delta

        diff2 = sum( ( p_n_p_1 - p_n ) ** 2 )
        if verbose: print '| p_n+1 - p_n |^2:', diff2
        p_n = p_n_p_1
        if diff2 < XSTEP_THRESHOLD: break

        if iteration >= MAX_ITERATIONS:
            print 'Diverged.'
            return p.copy()
            break

    return p_n.reshape( p.shape )

def compute_edge_lengths( p, edges ):
    '''
    Given a list of (i,j) pairs 'edges' denoting an edge between points p[i] and p[j],
    returns a list of rest lengths, one for each edge in 'edges'.

    NOTE: 'edges' must not have both (i,j) and (j,i)
    '''

    ## Check for duplicate edges, which are forbidden.
    edges = tuple( map( tuple, edges ) )
    from sets import ImmutableSet as Set
    assert len( Set( map( Set, edges ) ) ) == len( edges )

    result = []
    for i,j in edges:
        len_p_ij = sqrt( sum( (p[i] - p[j]) ** 2 ) )
        result.append( len_p_ij )

    return result

def test1():
    print '===== test1() ====='

    #p_undeformed = arrayf( [[0,0], [1,1]] )
    #p_undeformed = arrayf( [[0,0], [1,0], [1,1]] )
    #p_undeformed = arrayf( [[0,0], [1,0], [0,1], [1,1]] )
    p_undeformed = arrayf( [[0,0], [1,0], [2,0]] )
    print 'p.shape:', p_undeformed.shape
    print 'p undeformed:', p_undeformed

    #edges = [ (0,1) ]
    edges = [ (0,1), (1,2) ]
    #edges = [ (0,1), (0,2), (1,3), (2,3) ]
    print 'edges:', edges
    ## Multiply p_undeformed by 0 to force 0 rest length springs
    #edge_rest_lengths = compute_edge_lengths( 0 * p_undeformed, edges )
    edge_rest_lengths = compute_edge_lengths( p_undeformed, edges )
    print 'edge_rest_lengths:', edge_rest_lengths

    #constraints = [ ( 0, p_undeformed[0] ) ]
    #constraints = [ ( 0, p_undeformed[0] ), ( 3, p_undeformed[3] ) ]
    constraints = [ ( 0, p_undeformed[0] ), ( 2, p_undeformed[2] ) ]
    print 'constraints:', constraints

    p_initial = p_undeformed.copy()
    p_initial[1] += array( (.5,0) )
    print 'p initial:', p_initial
    p_solution = static_solution( p_initial, edges, edge_rest_lengths, constraints )
    print 'static solution:', p_solution

def main():
    test1()

if __name__ == '__main__': main()
