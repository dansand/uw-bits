from pylab import *
from myarray import *

import springs

def on_press(event):
    '''
    On mouse button down, if a point is close to the mouse, make it a constraint and drag the point.
    '''
    global motion_id, pts, down_pos, drag_ind, down_was_constraint
    prox_threshold = 0.05

    assert len(pts) > 0

    down_pos = ( event.xdata, event.ydata )
    if down_pos[0] is None or down_pos[1] is None: return

    # index of closest point on the list
    closest_ind = argmin(map( lambda z: norm( z - down_pos), pts ))
    drag_ind = -1

    # if there is a point close to the click point, drag it
    if norm( pts[closest_ind] - down_pos ) < prox_threshold:
        # enable callback for dragging
        motion_id = connect('motion_notify_event', on_motion)
        drag_ind = closest_ind

        pts[ drag_ind ] = down_pos
        down_was_constraint = drag_ind in constraints
        constraints[ drag_ind ] = down_pos

    update_points_and_edges()

def on_release(event):
    '''
    On mouse button release, disable motion callback.
    If the point was clicked and not dragged, toggle its constrained-ness.
    '''
    global motion_id, pts, drag_ind, down_pos, constraints
    click_threshold = 0.05

    if -1 != drag_ind and down_was_constraint and norm( pts[ drag_ind ] - down_pos ) < click_threshold:
        if True: #if len( constraints ) > 2:
            del constraints[ drag_ind ]
        else:
            print 'Removing the constraint will definitely lead to an underdetermined system. Constraint not removed.'

    drag_ind = -1
    disconnect(motion_id)
    update_points_and_edges()

def on_motion(event):
    '''
    On mouse drag, update the dragged point's constraint position, recompute the spring,
    and redraw the plot.
    '''
    global pts, drag_ind

    ## We shouldn't see this, but I don't want to assert because window systems do the
    ## darndest things.
    if drag_ind == -1: disconnect(motion_id)

    new_pt = ( event.xdata, event.ydata )
    if new_pt[0] is None or new_pt[1] is None: return

    constraints[ drag_ind ] = new_pt
    pts[ drag_ind ] = new_pt

    update_points_and_edges()

def update_points_and_edges():
    global pts

    ## Compute new locations for the points.
    pts = springs.static_solution( pts, edges, edges_rest_lengths, constraints.items(), verbose = False )

    ## Tell matplotlib about the new point positions.
    pts_un = pts.tolist()
    pts_c = []

    constraint_indices = constraints.keys()
    constraint_indices.sort()
    constraint_indices.reverse()
    for c in constraint_indices:
        del pts_un[ c ]
        pts_c.append( pts[ c ] )

    pts_un = asarrayf( pts_un )
    pts_c = asarrayf( pts_c )

    plt_un.set_data(pts.T[0],pts.T[1])
    if len( pts_c ): plt_c.set_data(pts_c.T[0],pts_c.T[1])
    else: plt_c.set_data([],[])


    ## Tell matplotlib about the new edge positions.
    for i, edge in enumerate( edges ):
        pt0 = pts[ edge[0] ]
        pt1 = pts[ edge[1] ]
        plt_edges[i].set_data( [ pt0[0], pt1[0] ], [ pt0[1], pt1[1] ] )

    draw()


pts = None
edges = None

def make_simple_chain():
    global pts, edges, constraints

    ## A super-simple chain of three nodes
    pts = arrayf( [ (-.5,0), (0,0), (.5,0) ] )
    edges = [ (0,1), (1,2) ]

    constraints = { 0: pts[0], 1: pts[1] }

def make_grid( n, m ):
    global pts, edges, constraints

    assert n > 1 and m > 1

    ## I want to make a grid of n-by-m points in the range [-.5, .5] x [-.5, .5].
    ## So a point i,j

    origin = arrayf( (-.5, -.5) )
    nstep = 1. / (n - 1)
    mstep = 1. / (m - 1)

    pts = []
    edges = []
    for i in xrange( n ):
        for j in xrange( m ):
            pts.append( origin + (j*mstep, i*nstep) )

            if i > 0: edges.append( ( i*m + j, (i-1)*m + (j-0) ) )
            if j > 0: edges.append( ( i*m + j, (i-0)*m + (j-1) ) )
            if j > 0 and i > 0:
                edges.append( ( (i-0)*m + j, (i-1)*m + (j-1) ) )
                edges.append( ( (i-1)*m + j, (i-0)*m + (j-1) ) )

    pts = arrayf( pts )

    constraints = { 0: pts[0], m-1: pts[m-1] }

#make_simple_chain()
#make_grid( 2, 3 )
make_grid( 4, 4 )


#edges_rest_lengths = springs.compute_edge_lengths( 0 * pts, edges )
edges_rest_lengths = springs.compute_edge_lengths( pts, edges )

drag_ind = -1
down_pos = (0,0)
down_was_constraint = True

plt_edges = [ plot([],'k-')[0] for e in edges ]
plt_un = plot([],'bo')[0]
plt_c = plot([],'ro')[0]

axis([-1,1,-1,1])

update_points_and_edges()

# initially, no callback for mouse motion
motion_id = 0
# callback for mouse button press and release
press_id = connect('button_press_event', on_press)
release_id = connect('button_release_event', on_release)

show()
