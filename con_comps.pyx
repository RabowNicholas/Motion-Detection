#Author: Nicholas Rabow
#Description: con_comps.pyx is cython file used to optimize connected components
#algorithm speed increases are to be expected by doing this. static typing and c
#functions are to be wrapped and performed in a multithreaded environment. This
#algorithm was attempted to be written in python first, but the implementation
#proved to slow. Research was done on how to improve the speed.
#See Wu et. al and Fiorio et. al. for algorithms implemented.
#Date: 12/9/2021
#Version: 1.1


#libraries
import numpy as np
from warnings import warn
cimport numpy as cnp
cnp.import_array()

#global variables
ctypedef cnp.intp_t DTYPE_t
DTYPE = np.intp
cdef DTYPE_t BG_NODE_NULL = -999 #node does not exist, used as placeholder

#objects
cdef struct s_shpinfo

#constructors
ctypedef s_shpinfo shape_info
ctypedef size_t (* fun_ravel)(size_t, size_t, size_t, shape_info *) nogil


######################
#BACKGROUND FUNCTIONS#
######################

# struct used concerning background in one place
ctypedef struct bginfo:
    DTYPE_t bg_val #The value in the image that identifies the background
    DTYPE_t bg_node #Node used to keep track of background
    DTYPE_t bg_label # Identification of the background in the label image

#Name: get_bginfo()
#Parameters: [bg_val]->value of background, *ret->used to update bginfo object
#Description: Update bginfo object with important information for PROCESSING
#Returns: updated bginfo object
cdef void get_bginfo(bg_val, bginfo *ret) except *:
    if bg_val is None:
        ret.bg_val = 0 #assume background value is 0
    else:
        ret.bg_val = bg_val
    ret.bg_node = BG_NODE_NULL  # The node -999 does not exist
    ret.bg_label = 0 #label 0 is the background

#Name: scanBG
#Parameters:[*data_p]->ptr to data_p array,[*forest_p]->ptr to forest_p array,
#           [*shapeinfo]->ptr to shapeinfor object, [*bg]->ptr to bginfo object
#Description: THIS IS A MUTEX LOCKED FUNCTION. All background pixels are dealt
#             with in this function. The purpose is to reduce unecessary background
#              scans.
#            This function updates forest_p and bg parameters inplace
cdef void scanBG((DTYPE_t *data_p, DTYPE_t *forest_p,
                  shape_info *shapeinfo,bginfo *bg)) nogil:
    cdef DTYPE_t i, bgval = bg.bg_val, firstbg = shapeinfo.numels #Variable definitions

    for i in range(shapeinfo.numels): #find all background pixels
        if data_p[i] == bgval:
            firstbg = i
            bg.bg_node = firstbg
            break

      #assign all background pixels to same label
    for i in range(firstbg, shapeinfo.numels):
        if data_p[i] == bgval:
            forest_p[i] = firstbg

#########################
#INPUT IMAGE INFORMATION#
#########################
# A pixel has neighbors that have already been scanned.
# In the paper, the pixel is denoted by E and its neighbors:
# in my code, E is shown with 0 and the neighbors(0-14):
# o_54 represents offset of 5 from 4
# used in get_shape_info function
cdef enum:
    o_54,o_51, o_52, o_53, o_57, o_59, o_510, o_511, o_513,o_COUNT

# Structure for centralized access to shape data
# Contains information related to the shape of the input array
cdef struct s_shpinfo:
    DTYPE_t x
    DTYPE_t y
    DTYPE_t numels #number of elements
    DTYPE_t ndim #dimensions of input array
    DTYPE_t off[o_COUNT] #offsets between elements
    fun_ravel ravel_index #fuction pointer to recalculate index as needed


#Name: get_shape_info
#Parameters: [f]->input image, [*res]->ptr to shape_info object
#Description: Precalculates all needed information about the input image shape
#             and stores it in shape_info object
cdef void get_shape_info(f_shape, shape_info *res) except *:
    res.x = f_shape[1]
    res.y = f_shape[0]
    res.ravel_index = res.x + res.y * res.x
    res.numels = res.x * res.y

    #offsets
    res.off[o_54] = -1
    res.off[o_51] = res.ravel_index(-1, -1, 0, res)
    res.off[o_52] = res.off[o_51] + 1
    res.off[o_53] = res.off[o_52] + 1

#################
#TREE OPERATIONS#
#################
# Tree operations implemented by an array as described in Wu et al.
# The term "forest" is used to indicate an array that stores one or more trees
# From paper:
# Consider a following tree:
#
# 5 ----> 3 ----> 2 ----> 1 <---- 6 <---- 7
#                 |               |
#          4 >----/               \----< 8 <---- 9
#
# The vertices are a unique number, so the tree can be represented by an
# array where a the tuple (index, array[index]) represents an edge,
# so for our example, array[2] == 1, array[7] == 6 and array[1] == 1, because
# 1 is the root.
# one array can hold more than one tree as long as their
# indices are different. It is the case in this algorithm, so for that reason
# the array is referred to as the "forest" = multiple trees next to each
# other.
#
# In this algorithm, there are as many indices as there are elements in the
# array to label and array[x] == x for all x. As the labelling progresses,
# equivalence between so-called provisional (i.e. not final) labels is
# discovered and trees begin to surface.
# When we found out that label 5 and 3 are the same, we assign array[5] = 3.

#Name:join_trees_wrapper
#Parameters: [*data_p]->ptr to image information,[*forest_p]->ptr to forest,
#            [rindex]->ravel index,[idxdiff]->offset index
#Description:Calls join trees function if necessary operation.
cdef inline void join_trees_wrapper(DTYPE_t *data_p, DTYPE_t *forest_p,
                                    DTYPE_t rindex, DTYPE_t idxdiff) nogil:
    if data_p[rindex] == data_p[rindex + idxdiff]:
        join_trees(forest_p, rindex, rindex + idxdiff)

#Name:find_root
#Parameters: [*forest]->ptr to forest, [n]->node
#Description: Find the root of node n.
cdef DTYPE_t find_root(DTYPE_t *forest, DTYPE_t n) nogil:
    cdef DTYPE_t root = n
    while (forest[root] < root):
        root = forest[root]
    return root

#Name: set_root
#Parameters: [*forest]->ptr to the forest, [n]->node, [root]t]
#Description: Sets all nodes on a path to point to new_root. Will eventually set
# all tree nodes to point to the real root.
cdef inline void set_root(DTYPE_t *forest, DTYPE_t n, DTYPE_t root) nogil:
    cdef DTYPE_t j
    while (forest[n] < n):
        j = forest[n]
        forest[n] = root
        n = j
    forest[n] = root

#Name: join_trees
#Parameters: [*forest]->ptr to forest, [n]->node,[m]->root
#Description: Join two trees containing nodes n and m.
cdef inline void join_trees(DTYPE_t *forest, DTYPE_t n, DTYPE_t m) nogil:
    cdef DTYPE_t root
    cdef DTYPE_t root_m

    if (n != m):
        root = find_root(forest, n)
        root_m = find_root(forest, m)

        if (root > root_m):
            root = root_m

        set_root(forest, n, root)
        set_root(forest, m, root)








# Flatten arrays are used to increase performance. Lookup is acheived by using
# precalculated offsets. Always starting at 5 and using this offset 'rindex' can
# be calculated which is the index of the pixel in the original image. offsets
# are located in shapeinfo object
#Name: scan2D
#Parameters: [*data_p]->ptr to input image pixels, [*forest]->ptr to forest,
#            [*shapeinfo]->ptr to shapeinfo obj, [*bg]->ptr to background info obj
#Description: Peforms forward scan on 2D array
cdef void scan2D(DTYPE_t *data_p, DTYPE_t *forest_p, shape_info *shapeinfo,
                 bginfo *bg) nogil:
    if shapeinfo.numels == 0:
        return
    cdef DTYPE_t x, y, rindex, bgval = bg.bg_val #store needed information
    cdef DTYPE_t *off = shapeinfo.off #array of offset values

    # Handle the first row
    for x in range(1, shapeinfo.x):
        rindex += 1
        if data_p[rindex] == bgval: # Nothing to do if we are background
            continue

        join_trees_wrapper(data_p, forest_p, rindex, off[o_54])
    for y in range(1, shapeinfo.y):
        # BEGINNING of x = 0
        rindex = shapeinfo.ravel_index(0, y, 0, shapeinfo)
        # Handle the first column
        if data_p[rindex] != bgval:
            # Nothing to do if we are background
            join_trees_wrapper(data_p, forest_p, rindex, off[o_52])
        # END of x = 0

        for x in range(1, shapeinfo.x - 1):
            # We have just moved to another column (of the same row)
            # so we increment the raveled index. It will be reset when we get
            # to another row, so we don't have to worry about altering it here.
            rindex += 1
            if data_p[rindex] == bgval: # Nothing to do if we are background
                continue

            join_trees_wrapper(data_p, forest_p, rindex, off[o_52])
            join_trees_wrapper(data_p, forest_p, rindex, off[o_54])
            join_trees_wrapper(data_p, forest_p, rindex, off[o_51])
            join_trees_wrapper(data_p, forest_p, rindex, off[o_53])

        # Finally, the last column
        # BEGINNING of x = max
        rindex += 1
        if data_p[rindex] != bgval: # Nothing to do if we are background

            join_trees_wrapper(data_p, forest_p, rindex, off[o_52])
            join_trees_wrapper(data_p, forest_p, rindex, off[o_54])
            join_trees_wrapper(data_p, forest_p, rindex, off[o_51])
        # END of x = max


######
#MAIN#
######
#Name: label_cython
#Parameters: [f]->input image
#Description: Cythonized version of label function from my original
#              implementation. Takes in image [f] and finds connected components
#               and labels them accordingly. This will be used to
#              find ball in images. Algorithm is describe in Fiorio et al.
#Returns: label image and number of labels
def label_cython(f, bg):
    #IMAGE INFORMATION
    f, swaps = reshape_array(f)
    shape = f.shape

    #OBJECT AND VARIABLE DECLARATION
    cdef cnp.ndarray[DTYPE_t, ndim=1] forest #disjoint union of trees
    out = np.array(f, order='C', dtype=DTYPE) #row major order
    forest = np.arange(data.size, dtype=DTYPE)

    cdef DTYPE_t *forest_p = <DTYPE_t*>forest.data #pointer to forest var
    cdef DTYPE_t *out_p = <DTYPE_t*>cnp.PyArray_DATA(out) #point to data

    cdef shape_info shapeinfo #shape_info structure used in algorithm
    cdef bginfo bg #background info structure used in algorithm

    get_shape_info(shape, &shapeinfo) #get input image info and stores
    get_bginfo(bg, &bg) #get background infor and store in [bg]

    # LABEL OUTPUT
    cdef DTYPE_t count #number of labeled components
    with nogil: #Global interpreter lock, mutex like funtionality
        scanBG(data_p, forest_p, &shapeinfo, &bg) #perform background scan
        scan2D(data_p, forest_p, &shapeinfo, &bg)
        count = resolve_labels(data_p, forest_p, &shapeinfo, &bg)

    return data, count

#Name: resolve_labels
#Parameters: [*data_p]->ptr to data_p,[*forest_p]->ptr to forest_p,
#            [*shapeinfo]->ptr to shapeinfo object, [*bg]->ptr to background obj
#Description: THIS IS MUTEX LOCKED FUNCTION
#             Second pass of algorithm to resolve any connected components that
#             are two different labels. Final labels are assigned and number of
#             labels are counted
cdef DTYPE_t resolve_labels(DTYPE_t *data_p, DTYPE_t *forest_p,
                            shape_info *shapeinfo, bginfo *bg) nogil:

    cdef DTYPE_t counter = 1, i #variable instantiation

    for i in range(shapeinfo.numels): #search every pixel
        if i == forest_p[i]: #root with new information
            if i == bg.bg_node: #root is background
                data_p[i] = bg.bg_label #assign to background
            else:
                data_p[i] = counter #otherwise assign to current label
                counter += 1
        else:
            data_p[i] = data_p[forest_p[i]] #pixel without new label information
    return counter - 1
