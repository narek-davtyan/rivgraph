# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:42:53 2018

@author: Jon
"""
import numpy as np
import itertools
from scipy.stats import mode
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from scipy.ndimage.morphology import distance_transform_edt
import rivgraph.ln_utils as lnu
import rivgraph.geo_utils as gu
import gdal
import pandas as pd


# Directionality algorithm numbering list:
# 1: set by inlets
# 2: set by outlets
# 3: set by continuity
# 4: set by knowledge of artificial node addition
# 5: set by

def plot_dirlinks(links, dims):

    def colorline(
        x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
            linewidth=3, alpha=1.0):
        """
        http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
        http://matplotlib.org/examples/pylab_examples/multicolored_line.html
        Plot a colored line with coordinates x and y
        Optionally specify colors in the array z
        Optionally specify a colormap, a norm function and a line width
        """
    
        # Default colors equally spaced on [0,1]:
        if z is None:
            z = np.linspace(0.0, 1.0, len(x))
    
        # Special case if a single number:
        if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
            z = np.array([z])
    
        z = np.asarray(z)
    
        segments = make_segments(x, y)
        if type(cmap) == str:
            lc = mcoll.LineCollection(segments, array=z, colors=cmap, norm=norm,
                                      linewidth=linewidth, alpha=alpha)
        else:
            lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                      linewidth=linewidth, alpha=alpha)
    
        ax = plt.gca()
        ax.add_collection(lc)
    
        return lc


    def make_segments(x, y):
        """
        Create list of line segments from x and y coordinates, in the correct format
        for LineCollection: an array of the form numlines x (points per line) x 2 (x
        and y) array
        """
    
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments


    fig, ax = plt.subplots()
    
    maxx = 0
    minx = np.inf
    maxy = 0
    miny = np.inf
    for l, certain in zip(links['idx'], links['certain']):
        if certain == 1:
            
            rc = np.unravel_index(l, dims)
        
            z = np.linspace(0, 1, len(rc[0]))    
            lc = colorline(rc[1], -rc[0], z, cmap=plt.get_cmap('cool'), linewidth=2)
        
            maxx = np.max([maxx, np.max(rc[1])])
            maxy = np.max([maxy, np.max(rc[0])])
            miny = np.min([miny, np.min(rc[1])])
            minx = np.min([minx, np.min(rc[0])])
    
    # Plot uncertain links
    for l, certain in zip(links['idx'], links['certain']):
        if certain != 1:
            
            rc = np.unravel_index(l, dims)
        
            z = np.linspace(0, 1, len(rc[0]))    
            lc = colorline(rc[1], -rc[0], z, cmap='white', linewidth=2)
        
            maxx = np.max([maxx, np.max(rc[1])])
            maxy = np.max([maxy, np.max(rc[0])])
            miny = np.min([miny, np.min(rc[1])])
            minx = np.min([minx, np.min(rc[0])])

    
    plt.show()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_facecolor("black")   
    plt.axis('equal')



def dir_inletoutlet(links, nodes):
    
    alg = 0
    
    # Set directionality of inlet links
    for i in nodes['inlets']:
        
        # Get links attached to inlets
        conn = nodes['conn'][nodes['id'].index(i)]
        
        for c in conn:
            linkidx = links['id'].index(c)
            
            # Set link directionality
            links, nodes = set_link(links, nodes, linkidx, i, alg=alg, checkcontinuity=True)
        
    # Set directionality of outlet links
    for o in nodes['outlets']:
        
        # Get links attached to outlets
        conn = nodes['conn'][nodes['id'].index(o)]
        
        for c in conn:
            linkidx = links['id'].index(c)
            
            # Set link directionality
            usnode = links['conn'][linkidx][:]
            usnode.remove(o)
            links, nodes = set_link(links, nodes, linkidx, usnode[0], alg=alg, checkcontinuity=True)
                                            
    return links, nodes



def dir_continuity(links, nodes, checknodes='all'):
    """
    Enforce continuity at each node; set link directionality where required.
    Iterates until no more links can be set.
    Can check only certain nodes using the 'checknodes' list.
    """
    alg = 1
    
    if checknodes == 'all':
        checknodes = nodes['id'][:]
    
    for nid in checknodes:
        nindex = nodes['id'].index(nid)
        nidx = nodes['idx'][nindex]
        conn = nodes['conn'][nindex]
                    
        # Initialize bookkeeping for all the links connected to this node
        linkdir = np.zeros((len(conn), 1), dtype=np.int) # 0 if uncertain, 1 if into, 2 if out of
        
        if linkdir.shape[0] < 2:
            continue
        
        # Populate linkdir
        for il, lid in enumerate(conn):
            lidx = links['id'].index(lid)
                                            
            # Determine if link is flowing into node or out of node
            # Skip if we're uncertain about the link's direction
            if links['certain'][lidx] == 0:
                continue
            elif links['idx'][lidx][0] == nidx: # out of
                linkdir[il] = 2
                
            elif links['idx'][lidx][-1] == nidx: # into
                linkdir[il] = 1
                
        if np.sum(linkdir==0) == 1: # If there is only a single unknown link
            unknown_link_id = conn[np.where(linkdir==0)[0][0]]
            unknown_link_idx = links['id'].index(unknown_link_id)
            m = mode(linkdir[linkdir>0])
            
            lconn = links['conn'][unknown_link_idx][:]
    
            if m.count[0] == linkdir.shape[0]-1: # if the non-zero elements are all the same (either all 1s or 2s)
                
                if m.mode[0] ==  1: # The unknown link must be out of the node
                    links, nodes = set_link(links, nodes, unknown_link_idx, nid, alg=alg)
                        
                elif m.mode[0] == 2: # The unknown link must be into the node
                    usnode = [n for n in lconn if n != nid][0]
                    links, nodes = set_link(links, nodes, unknown_link_idx, usnode, alg=alg)        
                                            
    return links, nodes



def dir_artificial_nodes(links, nodes, checknodes='all'):
    """
    Set the directionality of links where aritificial nodes were added. For such
    loops, flow will travel the same way through both sides of the loop (to avoid
    cycles). Therefore, if one side is known, we can set the other side.
    Method 1 sets a broken link if its counterpart is known.
    Method 2 sets a side of the loop if the other side is known.
    Method 3 sets both sides if the input to one of the end nodes is known.
    
    Can check chosen nodes by specifying their IDs in the checknodes list.
    """
    alg = 2
    for n in checknodes:
    
        # Determine if we're at a head node of an artificial loop
        nidx = nodes['id'].index(n)
        linkconn = nodes['conn'][nidx][:]
        for lc in linkconn:
            nodecheck = links['conn'][links['id'].index(lc)][:]
            nodecheck.remove(n)
            # If a neighboring node is an aritifical one, we're at a head node
            if nodecheck[0] in nodes['arts']:
                a_node = nodecheck[0]
                
                # Determine the short links
                shortlinks = nodes['conn'][nodes['id'].index(a_node)][:]
                endnodes = []
                for sl in shortlinks:
                    forappend = links['conn'][links['id'].index(sl)][:]
                    forappend.remove(a_node)
                    endnodes.append(forappend[0])

                # Find corresponding long link
                posslinks = nodes['conn'][nodes['id'].index(endnodes[0])]
                for p in posslinks:
                    linkidx = links['id'].index(p)
                    if set(links['conn'][linkidx]) == set(endnodes):
                        longlink = p
                        
                # Find link into head node to determine if its direction is known
                link_into = list(set(linkconn) - set([longlink]) - set(shortlinks))[0]
                longidx = links['id'].index(longlink)
                into_index = links['id'].index(link_into)
                
                # Check if link into head node direction is known and longlink is unknown
                if links['certain'][into_index] == 1 and links['certain'][longidx] == 0:
                    # Set longlink
                    if links['conn'][into_index][1] in links['conn'][longidx]:
                        usnode = links['conn'][into_index][1] 
                    else:
                        usnode = links['conn'][longidx][:]
                        usnode.remove(links['conn'][into_index][0])
                        usnode = usnode[0]
                    # Set the long link; shorter links of cycle should be set within the set_link function that recursively calls this one
                    links, nodes = set_link(links, nodes, longidx, usnode, alg=alg)        
                    
                    # Set short link that shares a node with longlink and link_into
                    shortlink = [l for l in shortlinks if n in links['conn'][links['id'].index(l)]][0]
                    slidx = links['id'].index(shortlink)
                    if links['conn'][into_index][1] in links['conn'][longidx]:
                        usnode = links['conn'][into_index][1] 
                    else:
                        usnode = links['conn'][slidx][:]
                        usnode.remove(links['conn'][into_index][0])
                        usnode = usnode[0]
                    # Set the long link; shorter links of cycle should be set within the set_link function that recursively calls this one
                    links, nodes = set_link(links, nodes, slidx, usnode, alg=alg)
                    
    return links, nodes



def dir_io_surface(links, nodes, dims):
    """
    Set directionality by first building a "DEM surface" where inlets are "hills"
    and outlets are "depressions," then setting links such that they flow
    downhill.
    """
    from scipy.stats import linregress
    from scipy.spatial import ConvexHull
    from scipy.interpolate import interp1d

    def hull_coords(xy):
        # Find the convex hull of a set of coordinates, then order them clockwisely
        # and remove the longest edge
        hull_verts = ConvexHull(np.transpose(np.vstack((xy[0], xy[1])))).vertices
        hull_coords = np.transpose(np.vstack((xy[0][hull_verts], xy[1][hull_verts])))
        hull_coords = np.reshape(np.append(hull_coords, [hull_coords[0,:]]), (int((hull_coords.size+2)/2), 2))
        # Find the biggest gap between hull points
        dists = np.sqrt((np.diff(hull_coords[:,0]))**2 + np.diff(hull_coords[:,1])**2)
        maxdist = np.argmax(dists) + 1
        first_part = hull_coords[maxdist:,:]
        second_part = hull_coords[0:maxdist,:]
        if first_part.size == 0:
            hull_coords = second_part
        elif second_part.size == 0:
            hull_coords = first_part
        else:
            hull_coords = np.concatenate((first_part, second_part))
            
        return hull_coords
    
    alg = 3

    # Create empty image to store surface
    I = np.zeros(dims, dtype=np.float) + 1
    
    # Get row,col coordinates of outlet nodes, arrange them in a clockwise order
    outs = [nodes['idx'][nodes['id'].index(o)] for o in nodes['outlets']]
    outsxy = np.unravel_index(outs, I.shape)
    hc = hull_coords(outsxy)

    # Burn the hull into the Iout surface
    for i in range(len(hc)-1):
        linterp = interp1d(hc[i:i+2,0], hc[i:i+2,1])  
        xinterp = np.arange(np.min(hc[i:i+2,0]), np.max(hc[i:i+2,0]), .1)
        yinterp = linterp(xinterp)
        for x,y in zip(xinterp,yinterp):
            I[int(round(x)), int(round(y))] = 0
            
    Iout = distance_transform_edt(I)
    Iout = (Iout - np.min(Iout)) / (np.max(Iout) - np.min(Iout))
    
    # Get coordinates of inlet nodes; use only the widest inlet and any inlets within 25% of its width
    ins = [nodes['idx'][nodes['id'].index(i)] for i in nodes['inlets']]
    in_wids = []
    for i in nodes['inlets']:
        linkid = nodes['conn'][nodes['id'].index(i)][0]
        linkidx = links['id'].index(linkid)
        in_wids.append(links['wid_adj'][linkidx])
    maxwid = max(in_wids)
    keep = [ii for ii, iw in enumerate(in_wids) if abs((iw - maxwid)/maxwid) < .25]
    ins_wide_enough = [ins[k] for k in keep]
    insxy = np.unravel_index(ins_wide_enough, dims)
    if len(insxy[0]) < 3:
        hci = np.transpose(np.vstack((insxy[0], insxy[1])))
    else:
        hci = hull_coords(insxy)
    
    # Burn the hull into the Iout surface
    I = np.zeros(dims, dtype=np.float) + 1
    if hci.shape[0] == 1:
        I[hci[0][0], hci[0][1]] = 0
    else:
        for i in range(len(hci)-1):
            linterp = interp1d(hci[i:i+2,0], hci[i:i+2,1])  
            xinterp = np.arange(np.min(hci[i:i+2,0]), np.max(hci[i:i+2,0]), .1)
            yinterp = linterp(xinterp)
            for x,y in zip(xinterp,yinterp):
                I[int(round(x)), int(round(y))] =0

    Iin = distance_transform_edt(I)
    Iin = np.max(Iin) - Iin
    Iin = (Iin - np.min(Iin)) / (np.max(Iin) - np.min(Iin))
    
    # Compute the final surface by adding the inlet and outlet images
    Isurf = Iout + Iin
    
#    lid = 650
#    linkidx = links['id'].index(lid)
#    rc = np.unravel_index(links['idx'][linkidx], dims)
#    plt.plot(rc[1], rc[0])
    
    # Determine the flow direction of each link
    slopes = []
    slopes2 = []
    for lid in links['id']:
                
        linkidx = links['id'].index(lid)
        lidcs = links['idx'][linkidx][:]
        rc = np.unravel_index(lidcs, dims)
        
        dists_temp = np.cumsum(np.sqrt(np.diff(rc[0])**2 + np.diff(rc[1])**2))
        dists_temp = np.insert(dists_temp, 0, 0)
        
        elevs = Isurf[rc[0], rc[1]]
        
        linreg = linregress(dists_temp, elevs)

        # Make sure slope is negative, else flip direction
        if linreg.slope > 0:
            usnode = nodes['id'][nodes['idx'].index(lidcs[-1])]
        else:
            usnode = nodes['id'][nodes['idx'].index(lidcs[0])]
                
        # Store guess
        links['guess'][linkidx].append(usnode)
        links['guess_alg'][linkidx].append(alg) 
        
        # Store slope
        slopes.append(linreg.slope)
        slopes2.append((elevs[-1]-elevs[0])/ dists_temp[-1])
        
    links['slope'] = slopes
    links['slope2'] = slopes2
        
    return links, nodes



def dir_main_channel(links, nodes, W_rel_to_L_weight=0.5):
    """
    Sets directionality of links based on "shortest" paths from widest inlet
    link to all the outlet links. Links are alsoe weighted by width, such that
    deviations from the "main channel" width cost more to traverse.
    
    W_rel_to_L_weight dictates how strong the weighting of width should be
    relative to length; higher values make width cost more (i.e. the shortest
    path will try to follow smaller widths rather than lengths).
    """
    
    alg = 4
    
    # Find widest inlet node
    inletW = [links['wid_pix'][links['id'].index(nodes['conn'][nodes['id'].index(nid)][0])][0] for nid in nodes['inlets']]
    W = max(inletW)
    inlet_idx = inletW.index(W)
    
    Lweight = links['len']
    Wweight = (W - links['wid_adj'])
    Wweight = Wweight * np.mean(Lweight) / np.mean(Wweight) * W_rel_to_L_weight
    # Give no cost to links with larger widths than the initial
    Wweight[Wweight<0] = 0
    
    weights = Lweight + Wweight
    
    # Create networkX graph, adding weighted edges
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], weights):
        G.add_edge(lc[0], lc[1], weight=wt)
    
    # Find shortest paths between all nodes
    allpaths = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))

    for o in nodes['outlets']:
    
        # Get the node-to-node path
        pathnodes = allpaths[nodes['inlets'][inlet_idx]][o]
    
        # Convert to link-to-link path
        pathlinks = []
        for u,v in zip(pathnodes[0:-1], pathnodes[1:]):
            ulinks = nodes['conn'][nodes['id'].index(u)]
            vlinks = nodes['conn'][nodes['id'].index(v)]
            pathlinks.append([ul for ul in ulinks if ul in vlinks][0])
                        
        # Set the directionality of each of the links
        for usnode, pl in zip(pathnodes, pathlinks):
        
            linkidx = links['id'].index(pl)
            
            # Don't set if already set
            if alg in links['guess_alg'][linkidx]:
                continue
            else:    
                # Store guess
                links['guess'][linkidx].append(usnode)
                links['guess_alg'][linkidx].append(alg) 
                
    return links, nodes



def dir_shortest_paths_nodes(links, nodes):
    """
    Determine link directionality based on the shortest path from its end
    nodes to the nearest outlet (or pre-outlet). If the path flows through
    the link attached to the node, its directionality is set; otherwise nothing
    is done. Note that this will not set all links' directionalities.
    """  
    alg = 5
    
    # Create networkX graph, adding weighted edges
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], links['len']):
        G.add_edge(lc[0], lc[1], weight=wt)
    
    # Get all "pre-outlet", i.e. nodes one link upstream of outlets. Use these so that decision of where to chop off outlet links doesn't play a role in shortest path.
    preoutlets = []
    for o in nodes['outlets']:
        linkconn = links['conn'][links['id'].index(nodes['conn'][nodes['id'].index(o)][0])]
        othernode = linkconn[:]
        othernode.remove(o)
        preoutlets.append(othernode[0])

    # All shortest paths
    msdpl = nx.multi_source_dijkstra_path(G, preoutlets)

    # Loop through all nodes; the directionality of the first link 
    # flowed through to reach the nearest outlet (or preoutlet) node is set
    for nid, nidx, nconn in zip(nodes['id'], nodes['idx'], nodes['conn']):
                        
        if nid in nodes['outlets'] or nid in nodes['inlets'] or nid in preoutlets:
            continue
        
        # Get the shortest path from nid to the nearest outlet or preoutlet        
        shortpath = msdpl[nid][::-1]
        
        # Set first link of the shortest path 
        # Find the link first
        for ip, posslink in enumerate(nconn):
            if set(links['conn'][links['id'].index(posslink)]) == set(shortpath[:2]):
                linkid = nconn[ip]
                
        linkidx = links['id'].index(linkid)
        
        if alg in links['guess_alg'][linkidx]:
            # If the guess agrees with previously guessed for this algorithm, move on
            if links['guess'][linkidx][links['guess_alg'].index(alg)] == nid:
                continue
            else:
                links['guess'][linkidx].remove(nid)
        else:
            # Update certainty
            links['guess'][linkidx].append(nid)
            links['guess_alg'][linkidx].append(alg) 
    
    return links, nodes



def dir_shortest_paths_links(links, nodes, difthresh = 0):
    """
    Loops through all links; determines a link's directionality by which of
    its endpoint nodes is closest to the nearest outlet (or preoutlet).
    "difthresh" refers to the difference in distances between endpoint nodes;
    higher means directionality is less likely to be ascertained.
    """
    alg = 6
    
    # Create networkX graph, adding weighted edges
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, wt in zip(links['conn'], links['len']):
        G.add_edge(lc[0], lc[1], weight=wt)
    
    # Get all "pre-outlet", i.e. nodes one link upstream of outlets. Use these so that decision of where to chop off outlet links doesn't play a role in shortest path.
    preoutlets = []
    for o in nodes['outlets']:
        linkconn = links['conn'][links['id'].index(nodes['conn'][nodes['id'].index(o)][0])]
        othernode = linkconn[:]
        othernode.remove(o)
        preoutlets.append(othernode[0])

    # NetworkX's multi-source dijkstra path length iterator        
    msdpl = nx.multi_source_dijkstra_path_length(G, preoutlets)

    # METHOD 2: Direction ascertained based on the nearness of a link's endnodes
    # to the nearest outlet [closer is downstream]
#    print(len(links['id']))
    for il,lid in enumerate(links['id']):
                
        linkidx = links['id'].index(lid)
        lconn = links['conn'][linkidx][:]
        
        # Compute shortest distance from each end node to nearest pre-outlet            
        node_min_len = []
        for endnode in lconn:
            node_min_len.append(msdpl[endnode])

        # Skip if the minima are too similar
        if abs(node_min_len[0] - node_min_len[1]) < difthresh:
            continue
        else: # DS node is the closer one
            dsnode = lconn[node_min_len.index(min(node_min_len))]
            usnode = list(set(lconn) - set([dsnode]))[0]
                    
            # Update certainty
            links['guess'][linkidx].append(usnode)
            links['guess_alg'][linkidx].append(alg) 
    
    return links, nodes



def dir_known_link_angles(links, nodes, dims, checklinks='all'):
    """
    Sets directionality for a link when all its immediately adjacent links'
    directionalities are known. Computes the angle of "flow" between outlet and
    inlet nodes for each neighboring link, then sets the unknown link
    such that its orientation minimizes the error between its neighbors and 
    itself.
    """
    alg=10

    def angle_between(v1, v2):
        """ 
        Returns the angle in degrees between vectors v1 and v2. Angle is computed
        in the clockwise direction from v1.
        """

        ang1 = np.arctan2(*v1[::-1])
        ang2 = np.arctan2(*v2[::-1])
        
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))

    
    if checklinks == 'all':
        checklinks = links['id']

    for lid in checklinks:
    
#        lid = 2212
        linkidx = links['id'].index(lid)
        conn = links['conn'][linkidx]
        lidcs = links['idx'][linkidx]
    
        # Ensure that all the directionalities of links connected to this one are known
        connlinks = set()
        for c in conn:
            linkconn = nodes['conn'][nodes['id'].index(c)][:]
            linkconn.remove(lid)
            connlinks.update(linkconn)
        
        certs = []
        for cl in connlinks:
            certs.append(links['certain'][links['id'].index(cl)])
            
        if sum(certs) != len(certs):
            continue
    
        # Coordinates of unknown link
        rc_idcs = np.unravel_index(lidcs, dims)
        
        # Get angles of all connected links (oriented up-to-downstream)
        angs = []
        horiz_vec = (1,0)
        for l in connlinks:
            linkidxt = links['id'].index(l)
            lidcst = links['idx'][linkidxt]
            rc = np.unravel_index(lidcst, dims)
            # Vector is downstream node minus upstream node
            ep_vec = (rc[1][-1]-rc[1][0], rc[0][0] - rc[0][-1])
            angs.append(angle_between(ep_vec, horiz_vec))
                
        # Compute angle for both orientations of unknown link
        poss_angs = []
        for orient in [0,1]:
            if orient == 0:
                ep_vec = (rc_idcs[1][-1]-rc_idcs[1][0], rc_idcs[0][0] - rc_idcs[0][-1])
            else:
                ep_vec = (rc_idcs[1][0]-rc_idcs[1][-1], rc_idcs[0][-1] - rc_idcs[0][0])
            poss_angs.append(angle_between(ep_vec, horiz_vec))
        
        # Compute the error
        err = [np.sqrt(np.sum((pa-angs)**2)) for pa in poss_angs]
        
        # Choose orientation with smallest error
        usnode = conn[err.index(min(err))]
        
        # Set the link
        links, nodes = set_link(links, nodes, linkidx, usnode, alg)
        
    return links, nodes



def dir_bridges(links, nodes):
    """
    Use shortest paths from bridge links to inlets/outlets to set link directions.
    """
    alg = 14
    
    # Create networkX graph object
    G = nx.Graph()
    G.add_nodes_from(nodes['id'])
    for lc, l in zip(links['conn'], links['len']):
        G.add_edge(lc[0], lc[1], weight=l)

    # Find bridge links; we don't want to count inlet and outlet links
    preoutlets = []
    for o in nodes['outlets']:
        linkconn = links['conn'][links['id'].index(nodes['conn'][nodes['id'].index(o)][0])]
        othernode = linkconn[:]
        othernode.remove(o)
        preoutlets.append(othernode[0])

    bridges = nx.bridges(G)
    bridges = list(bridges)
    bridgenodes = []
    endnodes = set(nodes['outlets']) | set(nodes['inlets']) | set(preoutlets)
    for b in bridges:
        bset = set(b) - endnodes
        if len(bset) == 2:
            bridgenodes.append(b)
            
    bridgelinks = []
    for bn in bridgenodes:
        conn = nodes['conn'][nodes['id'].index(bn[0])]
        for c in conn:
            if bn[1] in links['conn'][links['id'].index(c)]:
                bridgelinks.append(c)
                break

    # Set the bridge link
    # Remove the bridge link from the graph, then compare distances to reachable inlets/outlets to determine flow directions
    # Basically, determine which node of the bridge link is upstream and which is downstream
    for bl, bn in zip(bridgelinks, bridgenodes):
        G.remove_edge(bn[0], bn[1])
        # Group outlet nodes; those upstream and downstream of the removed bridge
        in_dists = []
        out_dists = []
        for bnode in bn:
            reachable = nx.single_source_dijkstra_path_length(G, bnode)
            d_ins_temp = []
            d_outs_temp = []
            for n in reachable:
                if n in nodes['outlets']:
                    d_outs_temp.append(reachable[n])
                elif n in nodes['inlets']:
                    d_ins_temp.append(reachable[n])
            in_dists.append(d_ins_temp)
            out_dists.append(d_outs_temp)
            
        # Create a distance matrix that describes the minimum distance to reachable inlets/outlets. nan values for where no inlets or outlets are reachable.
        dist_matrix = np.empty((2,2)) # rows are nodes in bn; columns are inlets, outlets; values are dists
        dist_matrix[:] = np.nan
        for i, idist in enumerate(in_dists): # inlets
            if len(idist) == 0:
                dist_matrix[i,0] = np.nan
            else:
                dist_matrix[i,0] = min(idist)
        for o, odist in enumerate(out_dists): # inlets
            if len(odist) == 0:
                dist_matrix[o,1] = np.nan
            else:
                dist_matrix[o,1] = min(odist)
                
        # Using the distance matrix, make decisions on which links to set
        nnans = np.sum(np.isnan(dist_matrix))
        if nnans == 2:  # Check for very unlikely case
            dsnode = np.nan
#            print('All flow must go through brige link {}; check that this is correct.'.format(bl))
        elif nnans == 1: # One node cannot reach any inlets or outlets, so we can guess its directions
            # Find the downstream and upstream nodes
            nanrow, nancol = np.where(np.isnan(dist_matrix))
            if nancol == 0:
                dsnode = bn[nanrow[0]]
            else:
                dsnode = list(set(bn) - set([bn[nanrow[0]]]))[0]
        else:
            dsnode = np.nan
        if dsnode != np.nan:
            usnode = list(set(bn) - set([dsnode]))[0]
        else:
            usnode = np.nan
        
        # Set the bridge link
        blidx = links['id'].index(bl)
        if usnode != np.nan:
            links['guess_alg'][blidx].append(alg)
            links['guess'][blidx].append(usnode)
            
        # Set the links down/upstream of the bridge link
        if nnans == 1:
            # Determine if we're routing to inlets or outlets
            if nancol[0] == 0: # setting outlets
                us_set = dsnode
                reachable = nx.descendants(G, us_set)
                ds_set = [r for r in reachable if r in nodes['outlets']]
            else: # setting inlets 
                ds_set = usnode
                reachable = nx.descendants(G, ds_set)
                us_set = [r for r in reachable if r in nodes['inlets']]
            
            if type(us_set) != list:
                us_set = [us_set]
            if type(ds_set) != list:
                ds_set = [ds_set]
                
            # Now we'll walk along each path and save the link's directionality
            for u in us_set:
                for d in ds_set:
                    pathnodes = nx.dijkstra_path(G, source=u, target=d)
                    # Convert to link-to-link path
                    pathlinks = []
                    for us,vs in zip(pathnodes[0:-1], pathnodes[1:]):
                        ulinks = nodes['conn'][nodes['id'].index(us)]
                        vlinks = nodes['conn'][nodes['id'].index(vs)]
                        pathlinks.append([ul for ul in ulinks if ul in vlinks][0])

                    for usnode, pl in zip(pathnodes, pathlinks):
                                
                        linkidx = links['id'].index(pl)
                        
                        # Add guess only if not already guessed OR if guessed in the other direction
                        n_already_guessed = links['guess_alg'][linkidx].count(alg)
                        if n_already_guessed == 0:
                            # Store guess
                            links['guess'][linkidx].append(usnode)
                            links['guess_alg'][linkidx].append(alg) 
                        elif n_already_guessed > 1:
                            continue
                        else: # Has been guessed once before; set only if this guess disagrees with earlier
                            if links['guess'][linkidx][links['guess_alg'][linkidx].index(alg)] != usnode:
                                # Store guess
                                links['guess'][linkidx].append(usnode)
                                links['guess_alg'][linkidx].append(alg) 
        
        # Add the bridge link back to the graph
        G.add_edge(bn[0], bn[1])
        
    # Now loop through and remove guesses where we've guessed both directions for the same link
    for lid, lga, lg in zip(links['id'], links['guess_alg'], links['guess']):
        if lga.count(alg) == 2:
            lidx = links['id'].index(lid)
            links['guess'][lidx] = [g for g, lgt in zip(lg, lga) if lgt != alg]
            links['guess_alg'][lidx] = [ga for ga in lga if ga != alg]
        
    return links, nodes



def set_link(links, nodes, linkidx, usnode, alg=9999, checkcontinuity=True):
    """
    Sets a link directionality; then checks for continuity and artificial nodes.
    """
    links['conn'][linkidx].remove(usnode)
    links['conn'][linkidx].insert(0,usnode)
    if links['idx'][linkidx][0] != nodes['idx'][nodes['id'].index(usnode)]:
        links['idx'][linkidx] = links['idx'][linkidx][::-1]
    
    links['certain'][linkidx] = 1
    links['certain_alg'][linkidx] = alg
    links['certain_order'][linkidx] = max(links['certain_order']) + 1
    
    if check_continuity is True:
        # Set any other possible links via continuity
        links, nodes = dir_continuity(links, nodes, checknodes=links['conn'][linkidx][:])
        # Set any other possible links via artificial nodes
        links, nodes = dir_artificial_nodes(links, nodes, checknodes=links['conn'][linkidx][:])

    return links, nodes


def fix_badnodes(links, nodes):
    """
    Fixes sources and sinks within the network by flipping link directionality.
    The link to flip is chosen by ensuring it does not create a cycle; if 
    multiple links can be flipped, the shortest one is chosen.
    """
    badnodes = check_continuity(links, nodes)
        
    for bn in badnodes:
        
        linkidx = None
        n_bn = len(check_continuity(links, nodes))
        
        # Get all the connected links
        lconn = nodes['conn'][nodes['id'].index(bn)]
                
        # Reverse their order and see if we've violated continuity or created a cycle
        bn_linkflip = [] # number of bad nodes after flipping link
        cycle_linkflip = []
        for l in lconn:
            lidx = links['id'].index(l)
            links['conn'][lidx] = links['conn'][lidx][::-1]
            links['idx'][lidx] = links['idx'][lidx][::-1]
          
            # See if we've violated continuity after flipping the link
            badnodes_temp = check_continuity(links, nodes)
            bn_linkflip.append(len(badnodes_temp))
            
            # See if we've created a cycle after flipping the link
            endnodes = links['conn'][lidx][:]
            c_nodes, _ = get_cycles(links, nodes, endnodes[0])
            c_nodes2, _ = get_cycles(links, nodes, endnodes[1])
            
            if c_nodes or c_nodes2:
                cycle_linkflip.append(1)
            else:
                cycle_linkflip.append(0)
                                    
            # Re-flip links to original position            
            links['conn'][lidx] = links['conn'][lidx][::-1]
            links['idx'][lidx] = links['idx'][lidx][::-1]
            
        # Now check if any of the flipped links solves the bad nodes AND doesn't
        # create a cycle--use that orientation if so
        poss_bn = [l for l, bnlf in zip(lconn, bn_linkflip) if bnlf + 1 == n_bn]
        poss_cy = [l for l, clf in zip(lconn, cycle_linkflip) if clf == 0]
        poss_links = list(set(poss_bn).intersection(set(poss_cy)))
        
        if len(poss_links) == 1: # Only one possible link we can flip, so do it
            linkidx = links['id'].index(poss_links[0])
        elif len(poss_links) == 0: # No possible links meet both criteria
            continue
#            if len(poss_bn) == 0:
#                print('Impossible to fix source/sink at node {}.'.format(bn))
#            else:
#                linklens = [links['len'][links['id'].index(l)] for l in poss_bn]
#                linkidx = links['id'].index(poss_bn[linklens.index(min(linklens))])
        else: # More than one link meets the criteria; choose the shortest
            linklens = [links['len'][links['id'].index(l)] for l in poss_links]
            linkidx = links['id'].index(poss_links[linklens.index(min(linklens))])
        
        if linkidx:
            set_link(links, nodes, linkidx, links['conn'][linkidx][1], alg=-2)

    return links, nodes



def check_continuity(links, nodes):
    """
    Check that there aren't any sinks or sources within the network, besides
    inlets and outlets.
    Returns any nodes where continuity is violated.
    """
    problem_nodes = []
    for nid, nidx, nconn in zip(nodes['id'], nodes['idx'], nodes['conn']):
        
        if nid in nodes['outlets'] or nid in nodes['inlets']:
            continue
        
        firstidx = []
        lastidx = []
        for linkid in nconn:
            linkidx = links['id'].index(linkid)
            firstidx.append(links['idx'][linkidx][0])
            lastidx.append(links['idx'][linkidx][-1])
            
        if firstidx[1:] == firstidx[:-1]:
            problem_nodes.append(nid)
        
        if lastidx[1:] == lastidx[:-1]:
            problem_nodes.append(nid)
            
    return problem_nodes


def get_cycles(links, nodes, checknode='all'):
    """
    Finds either all cycles in a graph or cycles containing the checknode'th 
    node. Cycles are returned as both nodes and links.
    """    
    G = nx.DiGraph()
    G.add_nodes_from(nodes['id'])
    for lc in links['conn']:
        G.add_edge(lc[0], lc[1])
        
    if checknode == 'all':
        cycle_nodes = nx.simple_cycles(G)
        # Unpack the iterator
        cycle_nodes = list(cycle_nodes)
    else:
        try:
            single_cycle = nx.find_cycle(G, source=checknode)
            single_cycle = list(single_cycle)
            cycle_nodes = []            
            for cn in single_cycle:
                cycle_nodes.append(cn[0])
            cycle_nodes = [cycle_nodes]
        except:
            cycle_nodes = None
        
    # Get links of cycles
    cycles_links = []        
    if cycle_nodes is not None:
        for c in cycle_nodes:
            pathlinks = []
            for us,vs in zip(c, c[1:] + [c[0]]):
                ulinks = nodes['conn'][nodes['id'].index(us)]
                vlinks = nodes['conn'][nodes['id'].index(vs)]
                pathlinks.append([ul for ul in ulinks if ul in vlinks][0])
            cycles_links.append(pathlinks)
    else:
        cycles_links = cycle_nodes
            
    return cycle_nodes, cycles_links


def flip_links_in_G(G, links2flip):
    """
    Flips the directionality of links in a networkx graph object G.
    links2flip is a N-elemnet tuple containing the US and DS nodes of the
    edge (link) to flip.
    """
    if links2flip == 'all':
        links2flip = list(G.edges)
    
    for lf in links2flip:
        # Remove the link
        G.remove_edge(lf[0], lf[1])
        # Reverse it and re-add it
        G.add_edge(lf[1], lf[0])
        
    return G

    

def fix_cycles(links, nodes):
    """
    This algorithm attempts to fix all cycles in the directed graph.
    """
    
    # Create networkx graph object
    G = nx.DiGraph()
    G.add_nodes_from(nodes['id'])
    for lc in links['conn']:
        G.add_edge(lc[0], lc[1])
        
    # Check for cycles
    if nx.is_directed_acyclic_graph(G) is not True:
        
        # Get list of cycles to fix
        c_nodes, c_links = get_cycles(links, nodes)
        
        # Remove any cycles that are subsets of larger cycles
        isin = np.empty((len(c_links),1))
        isin[:] = np.nan
        for icn, cn in enumerate(c_nodes):
            for icn2, cn2 in enumerate(c_nodes):
                if cn2 == cn:
                    continue
                elif len(set(cn) - set(cn2)) == 0:
                    isin[icn] = icn2
                    break
        cfix_nodes = [cn for icn, cn in enumerate(c_nodes) if np.isnan(isin[icn][0])]
        cfix_links = [cl for icl, cl in enumerate(c_links) if np.isnan(isin[icl][0])]

#        # Start with largest cycle
#        clen = [len(c) for c in c_nodes]
#        c_idx = clen.index(max(clen))
#        
#        cycle_n = c_nodes[c_idx]
#        cycle_l = c_links[c_idx]
        
        # Fix all the cycles
        for cycle_n, cycle_l in zip(cfix_nodes, cfix_links):
        
            # We will try every combination of flow directions and check each
            # combination for continuity violation and cycles, subsetting possibilities
            # to only those configurations that don't violate either.
            
            # First, subset our network to make computations faster; put subset
            # into networkX graph
            acl = []
            for nid in cycle_n:
                acl.extend(nodes['conn'][nodes['id'].index(nid)])
            all_cycle_links = set(acl)
            acn = []
            for lid in all_cycle_links:
                acn.extend(links['conn'][links['id'].index(lid)])
            all_cycle_nodes = set(acn)
            G = nx.DiGraph()
            G.add_nodes_from(all_cycle_nodes)
            for lid in all_cycle_links:
                lidx = links['id'].index(lid)
                lc = links['conn'][lidx]
                G.add_edge(lc[0], lc[1])           
                
            # Determine our "inlet" and "outlet" links/nodes - these we will not flip
            dangle_nodes = all_cycle_nodes - set(cycle_n)
            dangle_links = [l for l in all_cycle_links if len(set(links['conn'][links['id'].index(l)]) - dangle_nodes) == 1]
            ins_l = []
            outs_l = []
            ins_n = []
            outs_n = []
            dns = []
            for dl in dangle_links:
                lidx = links['id'].index(dl)
                lconn = links['conn'][lidx]
                dn = list(set(lconn).intersection(dangle_nodes))[0]
                dns.append(dn)
                if dn == lconn[0]:
                    ins_l.append(dl)
                    ins_n.append(dn)
                else:
                    outs_l.append(dl)
                    outs_n.append(dn)
#            if len(ins_l) == 0 or len(outs_l) == 0:
#                print('The cycle (links) {} appears to be either a sink or a source.'.format(cycle_l))
                
            # Try all configurations of links and count the number of cycles/continuity violations (don't change dangle links)
            # First, find all combinations
            fliplinks = list(all_cycle_links - set(dangle_links))
            all_combos = []
            for L in range(1, len(fliplinks)+1):
                for subset in itertools.combinations(fliplinks, L):
                    all_combos.append(subset)
                    
            # If a cycle is too big, there will be too many combinations to 
            # feasibly check all possibilities on a single processor. For now,
            # we just skip these and report them so they can be manually 
            # corrected.
            if len(all_combos) > 1024:
                print('The cycle links {} is too large to attempt to fix automatically.'.format(cycle_l))
                continue
                
            # Iterate through each combination and determine violations: there are four conditions that must be met:
            # 1) no cycles, 2) no sources/sinks, and 3) all inlets must be able to drain to an outlet, and all outlets must be reachable from at least one inlet
            # (Inlets and outlets refer to those of the subset, not the entire graph)
            # 4) links cannot flow against what has been manually set
            cont_violators = []
            len_cycle = []
            has_path = []
            manually_set = []
            for flink in all_combos:
    
                # Flip all the links in flink
                links2flip = []
                for fl in flink:
                    links2flip.append(links['conn'][links['id'].index(fl)][:])
                G = flip_links_in_G(G, links2flip)
    
                # Check if a cycle exists
                try:
                    cycles_temp = nx.find_cycle(G)
                    len_cycle.append(len(cycles_temp))
                except:
                    len_cycle.append(0)
                
                # Check if continuity is violated
                sink_nodes = set([node for node, outdegree in list(G.out_degree) if outdegree == 0]) - dangle_nodes
                source_nodes = set([node for node, indegree in list(G.in_degree) if indegree == 0]) - dangle_nodes
                cont_violators.append(len(sink_nodes) + len(source_nodes))
                
                # Check if each inflow can reach an outflow and each outflow is reached by an inflow
                hp_ins = []
                for ii in ins_n:
                    for oo in outs_n:
                        if nx.has_path(G, ii, oo) is True:
                            hp_ins.append(1)
                            break            
                # Flip links to test outlets
                G = flip_links_in_G(G, links2flip='all')
                hp_outs = []
                for oo in outs_n:
                    for ii in ins_n:
                        if nx.has_path(G, oo, ii) is True:
                            hp_outs.append(1)
                            break
                # Flip em back
                G = flip_links_in_G(G, links2flip='all')
    
                # Assign a "1" where inlet/outlet criteria are met
                if len(ins_n) == sum(hp_ins) and len(outs_n) == sum(hp_outs):
                    has_path.append(1)
                else:
                    has_path.append(0)
                
                # Check that we're not flipping any links that have been set manually
                set_by_alg = []
                for fl in flink:
                    set_by_alg.append(links['certain_alg'][links['id'].index(fl)])
                if -1 in set_by_alg:
                    manually_set.append(1)
                else:
                    manually_set.append(0)
                
                # Flip links back to original
                links2flipback = []
                for fl in flink:
                    c = links['conn'][links['id'].index(fl)][:]
                    links2flipback.append([c[1], c[0]])
                G = flip_links_in_G(G, links2flipback)
                    
            # Find configurations that don't violate continuity and have no cycles
            poss_configs = [i for i, (nv, nc, hp, ms) in enumerate(zip(cont_violators, len_cycle, has_path, manually_set)) if nv == 0 and nc == 0 and hp == 1 and ms == 0]
            
            if len(poss_configs) == 0:
                print('Unfixable cycle found at links: {}.'.format(cycle_l))
                continue
                        
            # Choose the configuration that flips the fewest links
            pc_lens = [len(all_combos[pc]) for pc in poss_configs]
            links_to_flip = all_combos[poss_configs[pc_lens.index(min(pc_lens))]]
            
            # Flip the links to fix the cycle
            for l in links_to_flip:
                lidx = links['id'].index(l)
                links['conn'][lidx] = links['conn'][lidx][::-1]
                links['idx'][lidx] = links['idx'][lidx][::-1]
                
    # Check if any cycles remain
    c_nodes, _ = get_cycles(links, nodes)
    if c_nodes is None:
        n_cycles_remaining = 0
    else:
        n_cycles_remaining = len(c_nodes)
                        
    return links, nodes, n_cycles_remaining


def guess_directionality(links, nodes, Idt, compute_guesses):
    """
    Using the links['guess'] dictionary, sets the directionality of all links.
    Directions are set in order of certainty--with the more certain being set
    first.
    """
        
    if compute_guesses != True and 'guess' not in links.keys():
        print('Direction guesses have not been computed yet; ignoring compute_guesses=False.')
        compute_guesses = True
    
    
    # Add a 'certain' entry to the links dict to keep track of if we're certain that
    # the direction has been set properly
    links['certain'] = np.zeros(len(links['id'])) # tracks whether a link's directinoality is certain or not
    links['certain_order'] = np.zeros(len(links['id'])) # tracks the order in which links certainty is set

    if compute_guesses is True:
        # Add widths and lengths to links
        links = lnu.link_widths_and_lengths(links, Idt)   

        links['certain_alg'] = np.zeros(len(links['id'])) # tracks the algorithm used to set certainty
        # Add a "guess" entry to keep track of the different guesses for flow directionality
        links['guess'] = [[] for a in range(len(links['id']))]
        links['guess_alg'] = [[] for a in range(len(links['id']))]
                 
        # Compute all the "guesses"
        links, nodes = dir_main_channel(links, nodes)
        links, nodes = dir_io_surface(links, nodes, Idt.shape)
        links, nodes = dir_shortest_paths_nodes(links, nodes)
        links, nodes = dir_shortest_paths_links(links, nodes)
        links, nodes = dir_bridges(links, nodes)
    
    # First, set inlet/outlet directions as they are 100% accurate    
    links, nodes = dir_inletoutlet(links, nodes)
    
#    # Filter the slopes; only keep those that are in the upper 3/4 quartile. Those
#    # near zero are unreliable.
#    slope75 = np.percentile(np.abs(links['slope']), 75)
#    for s, lid in zip(links['slope'], links['id']):
#        if abs(s) < slope75:
#            linkidx = links['id'].index(lid)
#            links['guess'][linkidx].pop(links['guess_alg'][linkidx].index(3))
#            links['guess_alg'][linkidx].remove(3)
        

    # Set the longest, steepest links according to io_surface (these are those we are most certain of)
    alg = 13
    len75 = np.percentile(links['len'], 75) 
    slope50 = np.percentile(np.abs(links['slope']), 50)
    for lid, llen, lg, lga, cert, lslope in zip(links['id'], links['len'], links['guess'], links['guess_alg'], links['certain'], links['slope']):
        if cert == 1:
            continue
        if llen > len75 and abs(lslope) > slope50:
            linkidx = links['id'].index(lid)
            if 3 in lga:
                usnode = lg[lga.index(3)]
                links, nodes = set_link(links, nodes, linkidx, usnode, alg)
   
    # Use bridges (14) to set links
    alg = 14
    for lid, idcs, lg, lga, cert in zip(links['id'], links['idx'], links['guess'], links['guess_alg'], links['certain']):
        # Only need to set links that haven't been set
        if cert == 1:
            continue
        linkidx = links['id'].index(lid)
        # Set all the links that are known from bridge links
        if alg in lga:
            links, nodes = set_link(links, nodes, linkidx, lg[lga.index(alg)], alg)

    # Use main channel (4) to set links
    alg = 4
    for lid, idcs, lg, lga, cert in zip(links['id'], links['idx'], links['guess'], links['guess_alg'], links['certain']):
        # Only need to set links that haven't been set
        if cert == 1:
            continue
        linkidx = links['id'].index(lid)
        # Set all the links that are known from main_channel
        if alg in lga:
            links, nodes = set_link(links, nodes, linkidx, lg[lga.index(alg)], alg)
            
    # If any three methods agree, set that link to whatever they agree on
    alg = 7
    for lid, idcs, lg, lga, cert in zip(links['id'], links['idx'], links['guess'], links['guess_alg'], links['certain']):
        # Only need to set links that haven't been set
        if cert == 1:
            continue
        linkidx = links['id'].index(lid)
        # Set all the links with 3 or more guesses that agree
        m = mode(lg) 
        if m.count[0] > 2:
            links, nodes = set_link(links, nodes, linkidx, m.mode[0], alg)
               
    # If two methods that are not shortest path (5 and 6) agree, set link
    alg = 8
    for lid, idcs, lg, lga, cert in zip(links['id'], links['idx'], links['guess'], links['guess_alg'], links['certain']):
        # Only need to set links that haven't been set
        if cert == 1:
            continue
        linkidx = links['id'].index(lid)
        # Set all the links with 2 or more same guesses that are not shortest path
        if 3 in lga and 5 in lga:
            if lg[lga.index(3)] == lg[lga.index(5)]:
                links, nodes = set_link(links, nodes, linkidx, lg[lga.index(3)], alg)
        elif 3 in lga and 6 in lga:
            if lg[lga.index(3)] == lg[lga.index(6)]:
                links, nodes = set_link(links, nodes, linkidx, lg[lga.index(3)], alg)
                
    # Use io_surface (3) to set links that are longer than the median link length
    alg = 9
    medlinklen = np.median(links['len']) 
    for lid, llen, lg, lga, cert in zip(links['id'], links['len'], links['guess'], links['guess_alg'], links['certain']):
        if cert == 1:
            continue
        if llen > medlinklen and 3 in lga:
            linkidx = links['id'].index(lid)
            usnode = lg[lga.index(3)]
            links, nodes = set_link(links, nodes, linkidx, usnode, alg)
            
    # Find remaining uncertain links
    uncertain = [l for l, lc in zip(links['id'], links['certain']) if lc != 1]
    
    # Use link angles to set as many of the remaining links as possible
    links, nodes = dir_known_link_angles(links, nodes, Idt.shape, checklinks=uncertain)
    
    # Find remaining uncertain links again
    uncertain = [l for l, lc in zip(links['id'], links['certain']) if lc != 1]
    
    # Set remaining uncertains according to io_surface (3)
    alg = 3
    for lid in uncertain:
        linkidx = links['id'].index(lid)
        if 3 in links['guess_alg'][linkidx]:
            usnode = links['guess'][linkidx][links['guess_alg'][linkidx].index(alg)]
            links, nodes = set_link(links, nodes, linkidx, usnode, alg)
                
    return links, nodes


   
def set_directionality(links, nodes, Imask, plot=True, compute_guesses=True, Idt=None):
          
    if Idt is None:
        Idt = distance_transform_edt(Imask)
    
    links, nodes = guess_directionality(links, nodes, Idt, compute_guesses=compute_guesses)
        
    # At this point, all links have been set. Check for nodes that violate 
    # continuity
    links, nodes = fix_badnodes(links, nodes)

    # Fix any cycles we may have created
    links, nodes, ncyc_remaining = fix_cycles(links, nodes)
    while ncyc_remaining > 0:
        old_n = ncyc_remaining
        links, nodes, ncyc_remaining = fix_cycles(links, nodes)
        if ncyc_remaining == old_n:
            if ncyc_remaining > 0:
                print('Not all cycles were resolved.')
            break

    # Report any nodes that need manual attention
    badnodes = check_continuity(links, nodes)
    if len(badnodes) > 0:
        print('Nodes {} violate continuity. Check nearby links and fix manually.'.format(badnodes))


#    # Save links and nodes with directionality
#    with open(paths['lnpickle'], 'wb') as f:
#        pickle.dump([links, nodes, dims], f)    
#

    if plot is True:
        plot_dirlinks(links, Imask.shape) 

    return links, nodes

#lid = 1513
#linkidx = links['id'].index(lid)
#print(links['certain_alg'][linkidx])
#print(links['certain_order'][linkidx])
#print(links['guess_alg'][linkidx])
#print(links['guess'][linkidx])
#print(links['slope'][linkidx])
#print(links['slope2'][linkidx])
#print(links['conn'][linkidx])
#
#plt.figure()
#plt.hist(links['certain_alg'], bins=np.arange(0,np.max(links['certain_alg']+2))-.5)
#
##plt.figure()
##plt.hist(links['slope'])            
#
#
#
#badnodes = check_continuity(links, nodes)
#    
#linktoflip = 744
#lidx = links['id'].index(linktoflip)
#links['idx'][lidx] = links['idx'][lidx][::-1]
#links['conn'][lidx] = links['conn'][lidx][::-1]
#
#
#cn, cl = get_cycles(links, nodes)
    
    
def set_dirs_manually(links, nodes, setlinks_csv):
    
    # Set any links manually
    links, nodes = dir_set_manually(links, nodes, setlinks_csv)

    # Ensure we haven't created any sinks/sources
    links, nodes = fix_badnodes(links, nodes)
    
    badnodes = check_continuity(links, nodes)
    if len(badnodes) > 0:
        print('Nodes {} still violate continuity. Check nearby links and re-fix manually.'.format(badnodes))

    # Fix any cycles we may have created
    links, nodes, ncyc_remaining = fix_cycles(links, nodes)
    while ncyc_remaining > 0:
        old_n = ncyc_remaining
        links, nodes, ncyc_remaining = fix_cycles(links, nodes)
        if ncyc_remaining == old_n:
            if ncyc_remaining > 0:
                _, cycle_links = get_cycles(links, nodes)
                print('The following cycles (returned as links) were not resolved: {}.'.format(cycle_links))
            break
        
    return links, nodes


def dir_set_manually(links, nodes, setlinks_csv):
    """
    Sets link directions based on a user-input csv-file. The csv file has 
    exactly two columns; one for the link id, and one for its upstream node.
    """
    alg = -1 
    
    # If needed, create a csv file for fixing link directions.
    df = pd.read_csv(setlinks_csv)
    
    # Check if any links have been manually corrected and correct them
    if len(df) != 0:
        usnodes = df['usnode'].values
        links_to_set = df['link_id'].values
        
        for lid, usn in zip(links_to_set, usnodes):
            links, nodes = set_link(links, nodes, links['id'].index(lid), usn, alg=alg)

    return links, nodes


