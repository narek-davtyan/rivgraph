# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:59:52 2018

@author: Jon
"""
import numpy as np
import rivgraph.geo_utils as gu
from scipy.stats import mode
from ordered_set import OrderedSet
import geopandas as gpd

def node_updater(nodes, idx, conn):
    
    """
    Updates node dictionary. Supply index of node and connected link.
    """
    
    if idx not in nodes['idx']:
        nodes['idx'].append(idx)
        
    if len(nodes['conn']) < len(nodes['idx']):
        nodes['conn'].append([])

    nodeid = nodes['idx'].index(idx)
    nodes['conn'][nodeid] = nodes['conn'][nodeid] + [conn]
        
    return nodes


def link_updater(links, linkid, idx=-1, conn=-1): 
    
    if linkid not in links['id']:
        links['id'].append(linkid)
        
    linkidx = links['id'].index(linkid)
    
    if idx != -1 :   
        if type(idx) is not list:
            idx = [idx]
            
        if len(links['idx']) < len(links['id']):
            links['idx'].append([])
            links['conn'].append([])
    
        links['idx'][linkidx] = links['idx'][linkidx] + idx
    
    if conn != -1:
        if len(links['conn']) < len(links['id']):
            links['conn'].append([])
            links['idx'].append([])
            
        links['conn'][linkidx] = links['conn'][linkidx] + [conn]
    
    return links


def delete_link(links, nodes, linkid):
    
    # Deletes linkid from links, updates nodes accordingly
    
    linkkeys = [lk for lk in links.keys() if type(links[lk]) is not int and len(links[lk]) == len(links['id'])]
    
#    linkkeys = links.keys() - set(['len', 'wid', 'wid_avg', 'n_networks'])
    
    lidx = links['id'].index(linkid)
        
    # Save the connecting nodes so we can update their connectivity ([:] makes a copy, not a view)
    connected_node_ids = links['conn'][lidx][:]
    
    # Remove the link and its properties
    for lk in linkkeys:
        if lk == 'id': # have to treat orderedset differently
            links[lk].remove(linkid)
        else:
            try:
                links[lk].pop(lidx)
            except:
                links[lk] = np.delete(links[lk], lidx)
            
    # Remove the link from node connectivity; delete nodes if there are no longer links connected
    for cni in connected_node_ids:
        cnodeidx = nodes['id'].index(cni)            
        nodes['conn'][cnodeidx].remove(linkid)   
        if len(nodes['conn'][cnodeidx]) == 0: # If there are no connections to the node, remove it 
            nodes = delete_node(nodes, cni)
            
    return links, nodes


def delete_node(nodes, nodeid, warn=True):
    # nodeid is the node ID, not 'idx'
    # A node should only be deleted after it no longer has any connected links
    # Delete a node and its properties
    
    nodekeys = nodes.keys() - set(['inlets', 'outlets', 'arts'])
    
    # Check that the node has no connectivity
    nodeidx = nodes['id'].index(nodeid)
    if len(nodes['conn'][nodeidx]) != 0 and warn==True:
        print('You are deleting node {} which still has connections to links.'.format(nodeid))
    
    # Remove the node and its properties
    for nk in nodekeys:
        if nk == 'id': # have to treat orderedset differently
            nodes[nk].remove(nodeid)
        elif nk == 'idx':
            nodes[nk].remove(nodes[nk][nodeidx])
        else:
            nodes[nk].pop(nodeidx)
            
    return nodes

 
def add_link(links, nodes, idcs):
    
    # Adds a new link using the indices found in idcs; nodes and connectivity
    # are also updated accordingly. Widths and lengths are not computed for
    # new links; must be recomputed for entire network
    
    # Find new link ID
    new_id = max(links['id']) + 1
        
    # Find connectivity of the new link to existing nodes
    lconn = []
    for lep in [idcs[0], idcs[-1]]:
        try:
            lconn.append(nodes['id'][nodes['idx'].index(lep)])
            nodes['conn'][nodes['idx'].index(lep)].append(new_id)
        except:
            # Add a new node if it's not found in the current ones
            nodes = add_node(nodes, lep, new_id)
            lconn.append(nodes['id'][nodes['idx'].index(lep)])

    if len(lconn) < 2:
        raise RuntimeError('Link is not connected to enough (2) nodes.')
    
    # Save new link
    links['conn'].append(lconn)
    links['id'].append(new_id)
    links['idx'].append(idcs)
        
    return links, nodes
                
        
def add_node(nodes, idx, linkconn):
    
    # Add a node to a set of links and nodes. Must provide the link ids connected
    # to the node (linkconn). Linkconn must be a list, even if only has one 
    # entry.
    
    if type(linkconn) is not list:
        linkconn = [linkconn]
    
    if idx in nodes['idx']:
        print('Node already in set; returning unchanged.')
        return nodes
    
    # Find new node ID
    new_id = max(nodes['id']) + 1
    
    # Append new node
    nodes['id'].append(new_id)
    nodes['idx'].append(idx)
    nodes['conn'].append(linkconn)
        
    return nodes


def link_widths_and_lengths(links, Idt):
    
    """
    Appends link widths and lengths to the links dictionary. 
    There is a slight twist. When a skeleton is computed for a very wide 
    channel with a narrow tributary, there is a very straight section of the 
    skeleton as it leaves the wide channel to go into the tributary; this 
    straight section should not be used to compute average link width, as it's 
    technically part of the wide channel. The twist here accounts for that by 
    elminating the ends of the each link from computing widths and lengths, 
    where the distance along each end is equal to the half-width of the endmost 
    pixels.    
    """
    
    # Compute and append link widths
    links['wid_pix'] = [] # width at each pixel
    links['len'] = []
    links['wid'] = []
    links['wid_adj'] = [] # average of all link pixels considered to be part of actual channel
    links['len_adj'] = []
    
    dims = Idt.shape
    
    for li in links['idx']:
        
        xy = np.unravel_index(li, dims)
        
        # Get widths at each pixel along each link
        widths = Idt[xy] * 2 # x2 because dt gives half-widths
        links['wid_pix'].append(widths)    
    
    # Compute the threshold number of pixels comprising a link required to use the trimmed link widths
    all_widths = [np.mean(wp) for wp in links['wid_pix']]
    mean_link_width = np.mean(all_widths)
    avg_thresh = int(mean_link_width/5)
    
    for li, widths in zip(links['idx'], links['wid_pix']):
        
        xy = np.unravel_index(li, dims)

        # Compute distances along link
        dists = np.cumsum(np.sqrt(np.diff(xy[0])**2 + np.diff(xy[1])**2))
        dists = np.insert(dists, 0, 0)
        
        # Compute distances along link in opposite direction
        revdists = np.cumsum(np.flipud(np.sqrt(np.diff(xy[0])**2 + np.diff(xy[1])**2)))
        revdists = np.insert(revdists, 0, 0)
        
        # Find the first and last pixel along the link that is at least a half-width's distance away
        startidx = np.argmin(np.abs(dists - widths[0]/2))
        endidx = len(dists) - np.argmin(np.abs(revdists - widths[-1]/2)) - 1
        
        # Need at least avg_thresh pixels to compute the adjusted average width; else use the entire link 
        if endidx - startidx < avg_thresh:
            links['wid_adj'].append(np.mean(widths))
            links['len_adj'].append(dists[-1])
        else:
            links['wid_adj'].append(np.mean(widths[startidx:endidx]))
            links['len_adj'].append(dists[endidx] - dists[startidx])
        
        # Unadjusted lengths and widths
        links['wid'].append(np.mean(widths))
        links['len'].append(dists[-1])
        
    return links


def conn_links(nodes, links, node_idx):
    """
    Returns the first and last pixels of links connected to node_idx.
    """
    
    link_ids = nodes['conn'][nodes['idx'].index(node_idx)]
    link_pix = []
    for l in link_ids:
        link_pix.extend([links['idx'][l][-1], links['idx'][l][0]])
        
    return link_pix


def adjust_for_padding(links, nodes, npad, dims, initial_dims):
    """
    Adjusts links['idx'] and nodes['idx'] values back to original image 
    dimensions, effectively removing the padding.
    """
    
    # Adjust the link indices
    adjusted_lidx = []
    for lidx in links['idx']:
        rc = np.unravel_index(lidx, dims)
        rc = (rc[0]-npad, rc[1]-npad)
        lidx_adj = np.ravel_multi_index(rc, initial_dims)
        adjusted_lidx.append(lidx_adj.tolist())
    links['idx'] = adjusted_lidx
        
    # Adjust the node idx
    adjusted_nidx = []
    for nidx in nodes['idx']:
        rc = np.unravel_index(nidx, dims)
        rc = (rc[0]-npad, rc[1]-npad)
        nidx_adj = np.ravel_multi_index(rc, initial_dims)
        adjusted_nidx.append(nidx_adj)
    nodes['idx'] = OrderedSet(adjusted_nidx)
        
    return links, nodes


def links_to_gpd(links, gdobj):
    """
    Converts links dictionary to a geopandas dataframe.
    """
    import shapely as shp
    from fiona.crs import from_epsg
        
    # Create geodataframe
    links_gpd = gpd.GeoDataFrame()
    
    # Assign CRS
    epsg = gu.get_EPSG(gdobj)
    links_gpd.crs = from_epsg(gu.get_EPSG(gdobj))
    
    # Append geometries
    geoms = []
    for i, lidx in enumerate(links['idx']):
        
        coords = gu.idx_to_coords(lidx, gdobj, inputEPSG=epsg, outputEPSG=epsg)
        geoms.append(shp.geometry.LineString(np.fliplr(coords))) 
    links_gpd['geometry'] = geoms
    
    # Append ids and connectivity
    links_gpd['id'] = links['id']
    links_gpd['us node'] = [c[0] for c in links['conn']]
    links_gpd['ds node'] = [c[1] for c in links['conn']]

    return links_gpd


def remove_all_spurs(links, nodes, dontremove=[]):
    # Remove links connected to nodes that have only one connection; this is 
    # performed iteratively until all spurs are removed. Spurs with inlet
    # nodes as endpoints are ignored.
    
    stopflag = 0
    while stopflag == 0:
        ct = 0
        # Remove spurs
        for nid, con in zip(nodes['id'], nodes['conn']):
            if len(con) == 1 and nid not in dontremove:
                ct = ct + 1
                links, nodes = delete_link(links, nodes, con[0])
                
        # Remove self-looping links (a link that starts and ends at the same node)
        for nid, con in zip(nodes['id'], nodes['conn']):
            m = mode(con)
            if m.count[0] > 1:
                
                # Get link 
                looplink = m.mode[0]                
                
                # Delete link
                links, nodes = delete_link(links, nodes, looplink)
                ct = ct + 1             
        
        
        # Remove all the nodes with only two links attached
        links, nodes = remove_two_link_nodes(links, nodes, dontremove)
        
        if ct == 0:
            stopflag = 1        
    
    return links, nodes


def remove_two_link_nodes(links, nodes, dontremove):
    # dontremove includes nodeids that should not be removed (inlets, outlets)
    
    # Removes unnecessary nodes
    linkkeys = [lk for lk in links.keys() if type(links[lk]) is not int and len(links[lk]) == len(links['id'])]
#    linkkeys = links.keys() - set(['len', 'wid', 'wid_avg', 'n_networks'])
    ct = 1
    while ct > 0:
        ct = 0
        for nidx, nid in enumerate(nodes['id']):
            # Get connectivity of current node        
            conn = nodes['conn'][nidx][:]
            # We want to combine links where a node has only two connections
            if len(conn) == 2 and nid not in dontremove:
                
                # Delete the node
                nodes = delete_node(nodes, nid, warn=False)
                
                # The first link in conn will be absorbed by the second
                lid_go = conn[0]
                lid_stay = conn[1]
                
                # Update the connectivity of the node attached to the link being absorbed
                conn_go = links['conn'][links['id'].index(lid_go)]
                conn_stay = links['conn'][links['id'].index(lid_stay)]
                
                # Update node connectivty of go link (stay link doesn't need updating)
                node_id_go = (set(conn_go) - set([nid])).pop()
                nodes['conn'][nodes['id'].index(node_id_go)].remove(lid_go)
                nodes['conn'][nodes['id'].index(node_id_go)].append(lid_stay)
                
                # Update link connectivity of stay link
                conn_go.remove(nid)
                conn_stay.remove(nid)
                # Add the "go" link connectivity to the "stay" link
                conn_stay.extend(conn_go)
                
                # Update the indices of the link
                idcs_go = links['idx'][links['id'].index(lid_go)]
                idcs_stay = links['idx'][links['id'].index(lid_stay)]
                if idcs_go[0] == idcs_stay[-1]:
                    new_idcs = idcs_stay[:-1] + idcs_go
                elif idcs_go[-1] == idcs_stay[0]:
                    new_idcs = idcs_go[:-1] + idcs_stay
                elif idcs_go[0] ==  idcs_stay[0]:
                    new_idcs = idcs_stay[::-1][:-1] + idcs_go
                elif idcs_go[-1] == idcs_stay[-1]:
                    new_idcs = idcs_stay[:-1] + idcs_go[::-1]
                links['idx'][links['id'].index(lid_stay)] = new_idcs
                
                # Delete the "go" link
                lidx_go = links['id'].index(lid_go)
                for lk in linkkeys:
                    if lk == 'id': # have to treat orderedset differently
                        links[lk].remove(lid_go)
                    else:
                        links[lk].pop(lidx_go)
                
                ct = ct + 1
                
    return links, nodes


def remove_single_pixel_links(links, nodes):
    """
    Thanks to the Lena for finding this bug. In very rare cases, we can end
    up with a link of one pixel. This function removes those.
    """
    # Find links to remove
    linkidx_remove = [lid for lidx, lid in zip(links['idx'], links['id']) if len(lidx) == 1]
    
    # Remove them
    for lidx in linkidx_remove:
        links, nodes = delete_link(links, nodes, lidx)
        
    return links, nodes


def append_link_lengths(links, gd_obj):
    
    epsg = gu.get_EPSG(gd_obj)
    
    # Compute and append link lengths -- assumes the CRS is in a projection that
    # respects distances
    links['len'] = []
    for idcs in links['idx']:
        link_coords = gu.idx_to_coords(idcs, gd_obj, inputEPSG=epsg, outputEPSG=epsg)
        dists = np.sqrt(np.diff(link_coords[:,0])**2 + np.diff(link_coords[:,1])**2)
        links['len'].append(np.sum(dists))
        
    return links


def add_artificial_nodes(links, nodes, gd_obj):
    
    # Add aritifical nodes to links that share the same two connected
    # nodes. This is written generally such that if there are more than two 
    # links that share endpoint nodes, aritifical nodes are added to all but
    # the shortest link. For simplicity of coding, when a node is added, the 
    # old link is deleted and two new links are put in its place.
    
    # Step 1. Find the link pairs that require artificial nodes
    # Put link conns into numpy array for sorting/manipulating
    link_conns = np.array(links['conn'])
    # Sort the link_conns
    link_conns.sort(axis=1)
    # Append the link ids to each row
    link_ids = np.expand_dims(np.array(links['id']),1)
    link_forsort = np.hstack((link_conns, link_ids))
    # Sort each row based on the first column
    link_forsort = link_forsort[link_forsort[:,0].argsort()]
    pairs = set()
    # This only checks for triplet-pairs. If there are four links that share the same two endpoint nodes, one of them will be missed. 
    for il in range(len(link_forsort)-2):
        if np.allclose(link_forsort[il, :2], link_forsort[il+1, :2]) and np.allclose(link_forsort[il, :2], link_forsort[il+2, :2]):
            pairs.add((link_forsort[il,2], link_forsort[il+1, 2], link_forsort[il+2, 2]))
        elif np.allclose(link_forsort[il, :2], link_forsort[il+1, :2]):
            pairs.add((link_forsort[il,2], link_forsort[il+1, 2]))
            
    # Extra lines to check ends that we missed
    if link_forsort[-2, 0:1] == link_forsort[-1 + 1, 0:1]:
        pairs.add((link_forsort[-2,2], link_forsort[-1, 2]))
    # Convert from set of tuples to list of lists
    pairs = [[*p] for p in pairs] # Pairs may also be triplets
    
    if 'len' not in links.keys():
        links = append_link_lengths(links, gd_obj)
    
    arts = []    
    # Step 2. Add the aritifical node to the proper links
    for p in pairs:

        # Choose the longest link(s) to add the artificial node
        lens = [links['len'][links['id'].index(l)] for l in p]
        minlenidx = np.argmin(lens)
        links_to_break = [l for il, l in enumerate(p) if il!=minlenidx]
        
        # Break each link and add a node
        for l2b in links_to_break:
    
            lidx = links['id'].index(l2b)
            idx = links['idx'][lidx]
            
            # Break link halfway; must find halfway first
            coords = gu.idx_to_coords(idx, gd_obj)
            dists = np.cumsum(np.sqrt(np.diff(coords[:,0])**2 + np.diff(coords[:,1])**2))
            dists = np.insert(dists, 0, 0)
            halfdist = dists[-1]/2
            halfidx = np.argmin(np.abs(dists-halfdist))
    
            # For simplicity, we will delete the old link and create two new links
            links, nodes = delete_link(links, nodes, l2b)
                
            # Create two new links
            newlink1_idcs = idx[:halfidx+1]
            newlink2_idcs = idx[halfidx:]
    
            # Adding links will also add the required artificial node
            links, nodes = add_link(links, nodes, newlink1_idcs)
            links, nodes = add_link(links, nodes, newlink2_idcs)
            
            arts.append(nodes['id'][nodes['idx'].index(idx[halfidx])])
            
    # Remove lengths from links
    _ = links.pop('len', None)
    
    # Store artificial nodes in nodes dict
    nodes['arts'] = arts
        
    return links, nodes




    
        