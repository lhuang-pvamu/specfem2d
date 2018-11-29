#!/usr/bin/env python2

import os.path
import sys
import numpy as np

class Model_spec(object):
    case = ""
    VpLayers = []
    VsLayers = []
    RhoLayers = []
    QkapLayers = []
    QmuLayers = []
    VpDict = {}
    
    def __init__(self,filename,sub):
    # Define parameters for the gridded model

        self.case = "KAUST Red Sea model"
        self.nzNodesIn = 2801
        self.zMax = 0.  # Z is elevation (increasing upwards), not depth
        self.dz = 1.25
        self.nxNodesIn = 9441
        self.xMin = 0.
        self.dx = 1.25
        self.nBdry = 5  # boundary layer thickness (cells)
        self.incX = sub
        self.incZ = sub

    # Layer properties dictionary
        # range of model P-velocities discretized to nearest 100
        self.VpLayers = np.multiply( range(14,54), 100.)
        self.nLayers = len(self.VpLayers)
        # other properties based on P-velocities
        self.VsLayers = np.round( np.multiply(0.806,self.VpLayers) - 856., 2)
        self.RhoLayers = np.round( 310. * np.power(self.VpLayers,0.25), 2)
        QpLayers = np.round( np.sqrt(self.VpLayers), 2)
        QsLayers = np.round( np.sqrt(self.VsLayers), 2)
        VsVpSq = np.power( np.divide(self.VsLayers,self.VpLayers), 2 )
        frac = np.subtract(1.0, VsVpSq)
        self.QkapLayers = np.round( np.multiply( QpLayers, frac), 2 )
        self.QmuLayers = QsLayers

        # Look-up table to get layer index from Vp value 
        self.VpDict = {}     
        for i in range(self.nLayers):
            self.VpDict[ np.int(self.VpLayers[i]) ] = i+1

        # Set prescribed Vs and Rho values for a few specific layers
        self.VsLayers[0] = 270.; self.RhoLayers[0] = 1750.  # overburden
        ixl = self.VpDict[ 4500 ] - 1
        self.VsLayers[ixl] = 2250.; self.RhoLayers[ixl] = 2170.     # salt
        ixl = self.VpDict[ 3900 ] - 1
        self.VsLayers[ixl] = 2050.; self.RhoLayers[ixl] = 2450.     # carbonates
        ixl = self.VpDict[ 5300 ] - 1
        self.VsLayers[ixl] = 3120.; self.RhoLayers[ixl] = 2650.     # basement

    # Derived values not specific to given model
        # Variables ending in Ext pertain to the "boundary-Extended" model
        self.nxCells = (self.nxNodesIn-1) / self.incX
        self.nzCells = (self.nzNodesIn-1) / self.incZ
        self.nCells = self.nxCells * self.nzCells
        self.nxNodes = self.nxCells+1
        self.nzNodes = self.nzCells+1
        self.nxNodesExt = self.nxNodes + 2*self.nBdry
        self.nzNodesExt = self.nzNodes + self.nBdry
        self.nNodesExt = self.nxNodesExt * self.nzNodesExt
        self.nxCellsExt = self.nxNodesExt-1
        self.nzCellsExt = self.nzNodesExt-1
        self.nCellsExt = self.nxCellsExt * self.nzCellsExt
        self.dx *= self.incX
        self.xMax = self.xMin + self.dx * self.nxCells
        self.xMinExt = self.xMin - self.dx * self.nBdry
        self.xMaxExt = self.xMax + self.dx * self.nBdry
        self.dz *= self.incZ
        self.zMin = self.zMax - self.dz * self.nzCells
        self.zMinExt = self.zMin - self.dz * self.nBdry

    # Open the P-velocity for for sequential access
        self.fdVp = open(filename,'r')
        self.inCount = 0

    def nextCol(self):
        # Return next sub-sampled velocity column
        if self.fdVp.closed:
            print "ERROR: Reading column",self.inCount,"beyond EOF on:",self.fdVp.name
            sys.exit(1)
        nzIn = self.nzNodesIn
        VpBuf = [0.]
        if self.inCount==0:
            self.inFirst = False
            VpBuf = np.fromfile(self.fdVp, dtype="f4", count=nzIn)
            self.inCount += 1
        else:
            for i in range(self.incX):  # take every incX columns
                VpBuf = np.fromfile(self.fdVp, dtype="f4", count=nzIn)
                if len(VpBuf) == 0:
                    self.fdVp.close()
                    return []
                self.inCount += 1        
        return VpBuf[ range(0,nzIn,self.incZ) ]     # take every incZ rows

    def __str__(self):
        return \
        "base model grid (%d X %d) nodes: \n" % (self.nxNodes,self.nzNodes) +\
        "x[i=%d]= %f, x[%d]= %f\n" % (self.nBdry,self.xMin,self.nxNodes-1+self.nBdry,self.xMax) +\
        "z[k=%d]= %f, z[%d]= %f\n" % (self.nBdry,self.zMin,self.nzNodes-1+self.nBdry,self.zMax) +\
        "extended model grid (%d X %d) nodes:\n" % (self.nxNodesExt,self.nzNodesExt) +\
        "x[i=%d]= %f, x[%d]= %f\n" % (0,self.xMinExt,self.nxNodesExt-1,self.xMaxExt) +\
        "z[k=%d]= %f, z[%d]= %f\n" % (0,self.zMinExt,self.nzNodesExt-1,self.zMax) 


    def cellNodes(self, im, km):
        # Return a list of 4 node indexes for the mesh element based at node (im,km)
        ixikLeft  = km + im*self.nzNodesExt
        ixikRight = km + (im+1)*self.nzNodesExt
        ibot = im + km*self.nxNodesExt
        itop = im + (km+1)*self.nxNodesExt
        return [ ixikLeft, ixikRight, ixikRight+1,ixikLeft+1 ]   # counter-clockwise

    def nodeXZ(self, i, k):
        # Return (x,z) for this grid node
        return [ self.xMinExt + i*self.dx, self.zMinExt + k*self.dz ]

    def ixProp(self, nodes, VpNodes):
        # Return the material property layer index for an element
        # defined by the given 4 nodes with given Vp values

        # We could get fancy here and do some spatial averaging based on
        # node positions and looking up in a more detailed Vp dictionary,
        # but for now we just take the mean of the four given Vp values  
        # and return the nearest "Layer" index from the dictionary.
        maxV = max(VpNodes)
        minV = min(VpNodes)
        avgV = np.round( sum(VpNodes)/len(VpNodes), -2)
        # if (maxV-avgV < avgV-minV):
        #     medV = maxV
        # else:
        #     medV = minV

        key = np.int(avgV)
        if key in self.VpDict:
            ixLayer = self.VpDict[ key ]
        else:
            print "ERROR in ixProp: Vp value",avgV,"not in dictionary!"
            ixLayer = 0
            sys.exit()
        return ixLayer

    
def grid_mesher(filename,sub):

    # Builds a SpecFEM2D "external mesh" using grid metadata,
    # with given material layer properties assigned to each cell (element)
    # according to P-velocity (Vp) values in a supplied gridded binary file.
    # The Vp(z,x) values are taken as assigned to nodes on the grid, and are 
    # arranged in vertical columns of increasing depth (more negative z-values)
    # below the surface at z=0.  Each cell is defined by four nodes, with median
    # value of the four material properties used to set the properties for the cell.

    # The input grid defines a "base" model which has a free surface on top,
    # and is padded on bottom and both sides to form an "extended" model with
    # absorbing boundaries. Indexing in the model grid starts at (i,k) = (0,0) at  
    # the bottom left end of the model.

    # We build the model from one column of the input grid at a time, 
    # flagging the top edge as the free surface, and the bottom edge values
    # extended into the lower absorbing boundary.
    # The first and last columns are treated specially for absorbing end layers.

    mdl = Model_spec(filename,sub)

    print mdl

    print "# Materials list for SpecFEM2D (viscoelastic)"
    print "nbmodels = ",mdl.nLayers
    for i in range(mdl.nLayers):
        layer = i+1
        print layer,1,mdl.RhoLayers[i],mdl.VpLayers[i],mdl.VsLayers[i],0,0,\
        mdl.QkapLayers[i],mdl.QmuLayers[i],0,0,0,0,0,0

    # Output files

    nodeCoordsFile = "mesh_node_XZ_coordinates"
    fdNodeCoords = open(nodeCoordsFile,'w')
    print >>fdNodeCoords, mdl.nNodesExt
    print "File:",nodeCoordsFile,"will have a line with (x,z) coordinates for each of",\
    mdl.nNodesExt,"nodes."

    # Write the node (x,z) coordinates
    for i in range(mdl.nxNodesExt):
        for k in range(mdl.nzNodesExt):
            x,z = mdl.nodeXZ(i, k)
            print >>fdNodeCoords, x, z 
    fdNodeCoords.close()
    print "grid_mesher: (X,Z) coordinates file written"

    cellNodesFile = "mesh_element_node_indices"
    fdCellNodes = open(cellNodesFile,'w')
    print >>fdCellNodes, mdl.nCellsExt
    print "File:",cellNodesFile,"will have a line with 4 indices for each of",\
    mdl.nCellsExt,"mesh elements."

    cellPropsFile = "mesh_element_property_indices"
    fdCellProps = open(cellPropsFile,'w')
    print "File:",cellPropsFile,"will have a line with one 'material' index for each of",\
    mdl.nCellsExt,"mesh elements."

    freeSurfFile = "free_surface_edges"
    fdFreeSurf = open(freeSurfFile,'w')
    print >>fdFreeSurf, mdl.nxCellsExt
    print "File:",freeSurfFile,"will have a line with two node indices for each of",\
    mdl.nxCellsExt,"free surface edges."

    absEdgeFile = "absorbing_boundary_edges"
    fdAbsEdge = open(absEdgeFile,'w')
    nEdges = mdl.nxNodes-1 + 2*(mdl.nzNodes-1)
    print >>fdAbsEdge, nEdges
    print "File:",absEdgeFile,"will have a line with two node indices for each of",\
    nEdges,"absorbing boundary edges."

    absCellFile = "absorbing_boundary_elements"
    fdAbsCells = open(absCellFile,'w')
    nBdryCells = ( mdl.nxCellsExt + 2*(mdl.nzNodes-1) )*mdl.nBdry
    print >>fdAbsCells, nBdryCells
    print "File:",absCellFile,"will have a line with element index and code for each of",\
    nBdryCells,"absorbing boundary elements."

    print "Reading columns from gridded binary Vp file:",filename

    # Process each column in extended model
    ixCell = 0      # count the cell elements as we create them
    isXbdry = False
    for ic in range(mdl.nxCellsExt):    # Process each column of cell elements
        if not isXbdry:
            VpBuf = mdl.nextCol()
            if len(VpBuf) > 0:
                VpCol = VpBuf
        if ic<mdl.nBdry or ic>=mdl.nxCellsExt-mdl.nBdry:
            print "boundary columns:",ic
            isXbdry = True
            VpPrev = VpCol

        # All columns
        km = mdl.nzNodesExt     # k-index in mesh counts downward
        for kr in range(mdl.nzNodes):
            km -= 1
            if kr==0:       # top edge -- record 2 nodes on free surface
                nodes = mdl.cellNodes(ic,km-1)
                print >>fdFreeSurf, ixCell+1, 2, nodes[2], nodes[3]
                continue

            ixCell += 1
            nodes = mdl.cellNodes(ic, km)
            print >>fdCellNodes, nodes[0],nodes[1],nodes[2],nodes[3]
            VpNodes = [ VpPrev[kr], VpCol[kr], VpCol[kr-1], VpPrev[kr-1] ]
            ixProp = mdl.ixProp(nodes, VpNodes)
            print >>fdCellProps, ixProp

            if ic==mdl.nBdry-1:     # left absorbing boundary edge
                print >>fdAbsEdge, ixCell, 2, nodes[1], nodes[2], 4
            if ic==mdl.nxCellsExt-mdl.nBdry: # right absorbing boundary edge
                print >>fdAbsEdge, ixCell, 2, nodes[3], nodes[0], 2

            if isXbdry:     # Record end boundary cells
                print >>fdAbsCells, ixCell, 1

        # at bottom of model, add boundary cells below
        while km > 0:
            km -= 1
            ixCell += 1
            Bflag = 2
            if isXbdry:
                Bflag = 3
            nodes = mdl.cellNodes(ic, km)
            print >>fdCellNodes, nodes[0],nodes[1],nodes[2],nodes[3]
            print >>fdAbsCells, ixCell, Bflag
            print >>fdCellProps, ixProp     # continue same material downward
            if km == mdl.nBdry-1:
              if ic>=mdl.nBdry and ic<mdl.nxCellsExt-mdl.nBdry:
                print >>fdAbsEdge, ixCell, 2, nodes[2], nodes[3], 1     # bottom edge

        if ic==mdl.nBdry-1:     # left absorbing boundary edge
            isXbdry = False
        VpPrev = VpCol

    fdCellNodes.close()
    fdCellProps.close()
    fdFreeSurf.close()
    fdAbsEdge.close()
    fdAbsCells.close()


def usage():
    print "usage: ./bin2mesh.py file [sub]"
    print "   where"
    print "       file - name of file containing grid of binary Vp velocity values"
    print "       sub  - subsample interval in Z and X (optional)"

if __name__ == '__main__':
    print "bin2mesh -- Create xmeshfem2d mesh files from binary data on (z,x) grid"
    print ""
    # get arguments
    if len(sys.argv) < 2:
        print "ERROR: 'file' argument needed"
        usage()
        sys.exit(1)
    file = sys.argv[1]
    sub = 1
    if len(sys.argv) >=2:
        sub = int(sys.argv[2])

    grid_mesher(file,sub)

