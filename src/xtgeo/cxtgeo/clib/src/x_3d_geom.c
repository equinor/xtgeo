/*
 ***************************************************************************************
 *
 * A collection of 3D geomtrical vectors, planes, etc
 *
 ***************************************************************************************
 */
#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_plane_normalvector.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Find the normal vector for a plane based on 3 points in 3D
 *    Based on: http://paulbourke.net/geometry/pointlineplane/
 *    The standard equation is Ax + By +Cz + D = 0 where (A,B,C)
 *    is the N vector
 *
 * ARGUMENTS:
 *    points_v       i     a [9] matrix with X Y Z of 3 points
 *    nvector        o     a [4] vector with A B C D
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: some input points are overlapping
 *              2: the input points forms a line
 *    Result nvector is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
int x_plane_normalvector(double *points_v, double *nvector, int option,
			 int debug)
{
    double x1, x2, x3, y1, y2, y3, z1, z2, z3, a, b, c, d;
    char  s[24]="x_plane_normalvector";

    xtgverbose(debug);

    xtg_speak(s,3,"Entering %s",s);

    x1 = points_v[0];
    y1 = points_v[1];
    z1 = points_v[2];

    x2 = points_v[3];
    y2 = points_v[4];
    z2 = points_v[5];

    x3 = points_v[6];
    y3 = points_v[7];
    z3 = points_v[8];

    /* some checks */
    if ((x1==x2 && y1==y2 && z1==z2) ||
	(x1==x3 && y1==y3 && z1==z3) ||
	(x3==x2 && y3==y2 && z3==z2)) {

	/* some points are the same */
	return(1);
    }


    a = y1*(z2-z3) + y2*(z3-z1) + y3*(z1-z2);
    b = z1*(x2-x3) + z2*(x3-x1) + z3*(x1-x2);
    c = x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2);
    d = -1*(x1*(y2*z3-y3*z2)+x2*(y3*z1-y1*z3)+x3*(y1*z2-y2*z1));


    if (a==0.0 && b==0.0 && c==0.0) {
	/* points on a line not forming a plane */
	return(2);
    }

    /* update nvector */
    nvector[0]=a;
    nvector[1]=b;
    nvector[2]=c;
    nvector[3]=d;

    return(0);
}


/*
 ***************************************************************************************
 *
 * NAME:
 *    x_isect_line_plane
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Finds the xyz coordinates where a line intersect a plane. The plane
 *    is infinite here, defined by its N vector. Based on:
 *    http://paulbourke.net/geometry/pointlineplane/
 *
 * ARGUMENTS:
 *    nvector        i     a [4] vector defining a N vector
 *    line_v         i     a [6] vector of two points (xyz xyz)
 *    point_v        o     a [3] vector with xyz of intersection point
 *    option         i     Options; use 2 if reurn 2 is needed to report
 *                         that line does not cross the plane
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *    Result point_v is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
int x_isect_line_plane(double *nvector, double *line_v, double *point_v,
		       int option, int debug)
{
    double x1, x2, y1, y2, z1, z2, a, b, c, d;
    double u, nom, dnom;
    char  s[24]="x_isect_line_plane";

    xtgverbose(debug);

    xtg_speak(s,3,"Enter %s",s);

    x1 = line_v[0];
    y1 = line_v[1];
    z1 = line_v[2];

    x2 = line_v[3];
    y2 = line_v[4];
    z2 = line_v[5];

    a = nvector[0];
    b = nvector[1];
    c = nvector[2];
    d = nvector[3];

    /* solve for u */
    nom  = (a*x1+b*y1+c*z1+d);
    dnom = (a*(x1-x2)+b*(y1-y2)+c*(z1-z2));

    if (fabs(dnom) < 0.0000000001) {
	/* the denominator is ~zero ... line is ~parallel to plane */
	return(1);
    }

    u = nom/dnom;

    if (option==2 && (u<0.0 || u>1.0)) {
	/* return 2 if the line does not cross the plane */
	return(2);
    }


    /* update intersection point */
    point_v[0] = x1 + u*(x2-x1);
    point_v[1] = y1 + u*(y2-y1);
    point_v[2] = z1 + u*(z2-z1);

    return(0);
}


/*
 ***************************************************************************************
 *
 * NAME:
 *    x_angle_vectors
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Find the angle in radians between two 3D vectors in space.
 *
 * ARGUMENTS:
 *    avec           i     The coeffs in A: ax + by + cz + d = 0
 *    bvec           i     The coeffs in B: ax + by + cz + d = 0
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Angle in radians
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
double x_angle_vectors(double *avec, double *bvec, int debug)
{
    char  s[24]="x_angle_vector";
    double dotproduct, maga, magb, angle;

    xtgverbose(debug);

    if (debug > 2) xtg_speak(s,3,"Enter %s",s);

    dotproduct = avec[0] * bvec[0] + avec[1] * bvec[1] + avec[2] * bvec[2];

    maga = sqrt(avec[0] * avec[0] + avec[1] * avec[1] + avec[2] * avec[2]);
    magb = sqrt(bvec[0] * bvec[0] + bvec[1] * bvec[1] + bvec[2] * bvec[2]);

    if (maga * magb < FLOATEPS) return 0.0;

    angle = acos(dotproduct / (maga * magb));

    return angle;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_sample_z_from_xy_cell
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes, May 2016
 *
 * DESCRIPTION:
 *    Given a XY point in 3D, does it intersect a cell top or base,
 *    and if so, what is the Z coordinate?
 *    This routine takes 4 points on 3D, forming a semi plane (typically
 *    a cell top). A the cell top is not a full plane, two triangle planes
 *    are computed, and the average is used
 *
 *    3      4       3      4
 *    --------       --------	   Two ways to divide a cell top into
 *    |     /|	     |\     |	   triangles
 *    |    / |	     | \    |
 *    |   /  |	     |  \   |
 *    |  /   |	     |   \  |
 *    | /    |	     |    \ |
 *    |/     |	     |     \|
 *    --------	     --------
 *    1       2      1       2
 *
 * ARGUMENTS:
 *    cell_v         i     a [24] vector defining xyz of each corner in a cell
 *    x              i     X coordinate
 *    y              i     Y coordinate
 *    option         i     0 for cell top, 1 for cell base
 *    option2        i     0 for avg in one trianglem 1 fro triangle1, 2 for triangle 2
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: Z value upon success. Else:
 *              UNDEF if XY outside cell top or base
 *              -UNDEF if other problems
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
double x_sample_z_from_xy_cell(double *cell_v, double x, double y,
                               int option, int option2, int debug)
{
    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    char   s[24]="x_sample_z_from_xy_cell";
    int    add, insidecell, inside, nfound, ier;
    double px[5], py[5], points[9], nvector1[4], nvector2[4], point_v[3];
    double line_v[6];
    double angle12, angle34, zloc1, zloc2, myzloc;

    xtgverbose(debug);

    //xtg_speak(s,3,"Into %s",s);

    /* make the XY point to a line in XYZ */
    line_v[0]=x;
    line_v[1]=y;
    line_v[2]=100;
    line_v[3]=x;
    line_v[4]=y;
    line_v[5]=1000;

    //xtg_speak(s,3,"Hmmm");

    zloc1 = UNDEF;
    zloc2 = UNDEF;

    //xtg_speak(s,3,"Evaluate %f %f ...",x,y);

    add=0;
    if (option==1) add=12;

    x1 = cell_v[add+0];
    y1 = cell_v[add+1];
    z1 = cell_v[add+2];

    x2 = cell_v[add+3];
    y2 = cell_v[add+4];
    z2 = cell_v[add+5];

    x3 = cell_v[add+6];
    y3 = cell_v[add+7];
    z3 = cell_v[add+8];

    x4 = cell_v[add+9];
    y4 = cell_v[add+10];
    z4 = cell_v[add+11];

    /* first see if point is inside the polygon formed by the cell corners */
    /* must be ordered clock or anticlock wise */
    px[0]=x1; px[1]=x2; px[2]=x4; px[3]=x3; px[4]=x1;
    py[0]=y1; py[1]=y2; py[2]=y4; py[3]=y3; py[4]=y1;



    insidecell = pol_chk_point_inside(x,y,px,py,5,debug);

    /* accept both inside (2) or edge (1): */
    if (insidecell < 1) {
	return(UNDEF);
    }

    /* OK, points is inside plane, now evaluate all triangles */

    /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    /* (1) triangle 1 --> 4 --> 3  */
    /* (2) triangle 1 --> 2 --> 4  */

    nfound=0;

    px[0]=x1; px[1]=x4; px[2]=x3; px[3]=x1;
    py[0]=y1; py[1]=y4; py[2]=y3; py[3]=y1;

    inside = pol_chk_point_inside(x, y, px, py, 4, debug); /* check for triangle */

    points[0]=x1; points[1]=y1; points[2]=z1;
    points[3]=x4; points[4]=y4; points[5]=z4;
    points[6]=x3; points[7]=y3; points[8]=z3;

    ier = x_plane_normalvector(points, nvector1, 0, debug);
    if (ier != 0) xtg_error(s,"Unforseen problems; report bug");

    if (inside > 0) {
	nfound=1;
	/* find the intersection */
	ier = x_isect_line_plane(nvector1, line_v, point_v, 0, debug);
	if (ier != 0) xtg_error(s,"Unforseen problems; report bug");
	zloc1 = point_v[2];
    }
    else{
	nfound=0;
    }

    /* check the other triangle */
    px[0]=x1; px[1]=x2; px[2]=x4; px[3]=x1;
    py[0]=y1; py[1]=y2; py[2]=y4; py[3]=y1;

    inside = pol_chk_point_inside(x,y,px,py,4,debug); /* check for triangle */
    points[0]=x1; points[1]=y1; points[2]=z1;
    points[3]=x2; points[4]=y2; points[5]=z2;
    points[6]=x4; points[7]=y4; points[8]=z4;

    ier = x_plane_normalvector(points, nvector2, 0, debug);

    if (nfound==0 && inside > 0) {
	nfound=1;
	/* find the intersection */
	ier = x_isect_line_plane(nvector2, line_v, point_v, 0, debug);
	if (ier != 0) xtg_error(s,"Unforseen problems; report bug");
	zloc1 = point_v[2];
    }

    angle12 = x_angle_vectors(nvector1, nvector2, debug);

    /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    /* (1) triangle 1 --> 2 --> 3  */
    /* (2) triangle 2 --> 3 --> 4  */

    nfound=0;

    px[0]=x1; px[1]=x2; px[2]=x3; px[3]=x1;
    py[0]=y1; py[1]=y2; py[2]=y3; py[3]=y1;

    inside = pol_chk_point_inside(x,y,px,py,4,debug); /* check for triangle */

    points[0]=x1; points[1]=y1; points[2]=z1;
    points[3]=x2; points[4]=y2; points[5]=z2;
    points[6]=x3; points[7]=y3; points[8]=z3;

    /* find normal vector */
    ier = x_plane_normalvector(points, nvector1, 0, debug);
    if (ier != 0) xtg_error(s,"Unforseen problems; report bug");

    if (inside > 0) {
	nfound=1;
	/* find the intersection */
	ier = x_isect_line_plane(nvector1, line_v, point_v, 0, debug);
	if (ier != 0) xtg_error(s,"Unforseen problems; report bug");
	zloc2 = point_v[2];
    }
    else{
	nfound=0;
    }

    /* check the other triangle */
    px[0]=x2; px[1]=x3; px[2]=x4; px[3]=x2;
    py[0]=y2; py[1]=y3; py[2]=y4; py[3]=y2;

    inside = pol_chk_point_inside(x,y,px,py,4,debug); /* check for triangle */

    points[0]=x2; points[1]=y2; points[2]=z2;
    points[3]=x3; points[4]=y3; points[5]=z3;
    points[6]=x4; points[7]=y4; points[8]=z4;

    /* find normal vector */
    ier = x_plane_normalvector(points, nvector2, 0, debug);
    if (ier != 0) xtg_error(s,"Unforseen problems; report bug");

    if (nfound==0 && inside > 0) {
	nfound=1;
	/* find the intersection */
	ier = x_isect_line_plane(nvector2, line_v, point_v, 0, debug);
	if (ier != 0) xtg_error(s,"Unforseen problems; report bug");
	zloc2 = point_v[2];
    }

    angle34 = x_angle_vectors(nvector1, nvector2, debug);


    if (zloc1 > UNDEF_LIMIT && zloc2 < UNDEF_LIMIT ) {
	xtg_error(s,"Something fishy ZLOC1 is undef while not ZLOC2: %f vs %f",
		  zloc1, zloc2);
    }
    else if (zloc1 < UNDEF_LIMIT && zloc2 > UNDEF_LIMIT ) {
	xtg_error(s,"Something fishy ZLOC2 is undef while not ZLOC1: %f vs %f",
		  zloc2, zloc1);
    }

    /* the final Z coordinate is as default the average of zloc1 and zloc2
     * this is really tricky and compared with RMS. It is NONTRIVIAL and grid tops
     * are not planes but curves, but all in all it seems that option2 = 0
     * (avg) is the best choice.
     */

    myzloc = 0.5 * (zloc1 + zloc2);

    if (option2 == 1) {
        myzloc = zloc1;
    }
    else if (option2 == 2) {
        myzloc = zloc2;
    }
    else if (option2 == 3) {
        if (angle12 < angle34) myzloc = zloc1;
        if (angle12 >= angle34) myzloc = zloc2;
    }
    else if (option2 == 4) {
        if (angle12 >= angle34) myzloc = zloc1;
        if (angle12 < angle34) myzloc = zloc2;
    }

    return(myzloc);
}
