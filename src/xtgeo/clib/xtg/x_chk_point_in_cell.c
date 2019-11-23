#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"
/*
 *=============================================================================
 * Check if a point is within a crid cell. The cell has the same organization
 * as Eclipse cells. This routine is adapted from Eclpost (kvpvos.f)
 * If point is within cell, a number > 0 is returned
 * The array coor as 24 entries (0..23) and a point is found by:
 * X,Y,Z for Corner 7 = coor[3*7-4+j], j=1 for X, 2 for Y and 3 for Z
 * JCR 18-JAN-2001
 *=============================================================================
 */


int x_chk_point_in_cell (
			 double x,
			 double y,
			 double z,
			 double coor[],
			 int   imethod,
			 int   debug
			 )
{

    double pp[3], pm[3], tri[4][3];
    int   istat[13], ier;
    double cbig=1.0e14, vminx, vmaxx, vminy, vmaxy, vminz, vmaxz;
    int   i, ic, isum;

    /*
     * Initialize
     */



    istat[0]=0;
    /*cbig=1e14;*/

    ier=0;
    pp[0]=x;
    pp[1]=y;
    pp[2]=z;


    /*
     * Check if point is outside cube
     */
    vminx=cbig;
    vmaxx=-cbig;
    vminy=cbig;
    vmaxy=-cbig;
    vminz=cbig;
    vmaxz=-cbig;

    for (i=1;i<=8;i++) {
	if (vminx > coor[3*i-3]) vminx=coor[3*i-3];
	if (vmaxx < coor[3*i-3]) vmaxx=coor[3*i-3];
	if (vminy > coor[3*i-2]) vminy=coor[3*i-2];
	if (vmaxy < coor[3*i-2]) vmaxy=coor[3*i-2];
	if (vminz > coor[3*i-1]) vminz=coor[3*i-1];
	if (vmaxz < coor[3*i-1]) vmaxz=coor[3*i-1];
    }

    if (x < vminx) return 0;
    if (x > vmaxx) return 0;
    if (y < vminy) return 0;
    if (y > vmaxy) return 0;
    if (z < vminz) return 0;
    if (z > vmaxz) return 0;

    /*
     * Unfortunenately, the point may lie outside cube even if the test above fails...:
     */


    /*
     * --------------------------------------------------------------------------
     * Tetrahedrons...
     * Cell midpoint pm[0]~X pm[1]~Y pm[2]~Z
     * --------------------------------------------------------------------------
     */
    pm[0]=0.0;
    pm[1]=0.0;
    pm[2]=0.0;

    for (ic=1;ic<=8;ic++) {
	for (i=0;i<3;i++) {
	    pm[i]=pm[i]+0.125*coor[3*ic-3+i];
	}
    }


    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 1
     * There are four corners made up of edges and midpoint
     * Corner 1 3 7 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*1 - 3 + i];
	tri[1][i] = coor[3*3 - 3 + i];
	tri[2][i] = coor[3*7 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[1]=x_kvpt3s(pp,tri,&ier);

    if (istat[1] == 2) {
	return istat[1];
    }


    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 2
     * There are four corners made up of edges and midpoint
     * Corner 1 5 7 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*1 - 3 + i];
	tri[1][i] = coor[3*5 - 3 + i];
	tri[2][i] = coor[3*7 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[2]=x_kvpt3s(pp,tri,&ier);
    if (istat[2] == 2) {
	return istat[2];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 3
     * There are four corners made up of edges and midpoint
     * Corner 1 5 6 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*1 - 3 + i];
	tri[1][i] = coor[3*5 - 3 + i];
	tri[2][i] = coor[3*6 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[3]=x_kvpt3s(pp,tri,&ier);
    if (istat[3] == 2) {
	return istat[3];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 4
     * There are four corners made up of edges and midpoint
     * Corner 1 2 6 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*1 - 3 + i];
	tri[1][i] = coor[3*2 - 3 + i];
	tri[2][i] = coor[3*6 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[4]=x_kvpt3s(pp,tri,&ier);
    if (istat[4] == 2) {
	return istat[4];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 5
     * There are four corners made up of edges and midpoint
     * Corner 2 4 8 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*2 - 3 + i];
	tri[1][i] = coor[3*4 - 3 + i];
	tri[2][i] = coor[3*8 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[5]=x_kvpt3s(pp,tri,&ier);
    if (istat[5] == 2) {
	return istat[5];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 6
     * There are four corners made up of edges and midpoint
     * Corner 2 6 8 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*2 - 3 + i];
	tri[1][i] = coor[3*6 - 3 + i];
	tri[2][i] = coor[3*8 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[6]=x_kvpt3s(pp,tri,&ier);
    if (istat[6] == 2) {
	return istat[6];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 7
     * There are four corners made up of edges and midpoint
     * Corner 3 4 8 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*3 - 3 + i];
	tri[1][i] = coor[3*4 - 3 + i];
	tri[2][i] = coor[3*8 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[7]=x_kvpt3s(pp,tri,&ier);
    if (istat[7] == 2) {
	return istat[7];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 8
     * There are four corners made up of edges and midpoint
     * Corner 3 7 8 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*3 - 3 + i];
	tri[1][i] = coor[3*7 - 3 + i];
	tri[2][i] = coor[3*8 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[8]=x_kvpt3s(pp,tri,&ier);
    if (istat[8] == 2) {
	return istat[8];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 9
     * There are four corners made up of edges and midpoint
     * Corner 1 2 3 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*1 - 3 + i];
	tri[1][i] = coor[3*2 - 3 + i];
	tri[2][i] = coor[3*3 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[9]=x_kvpt3s(pp,tri,&ier);
    if (istat[9] == 2) {
	return istat[9];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 10
     * There are four corners made up of edges and midpoint
     * Corner 2 3 4 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*2 - 3 + i];
	tri[1][i] = coor[3*3 - 3 + i];
	tri[2][i] = coor[3*4 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[10]=x_kvpt3s(pp,tri,&ier);
    if (istat[10] == 2) {
	return istat[10];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 11
     * There are four corners made up of edges and midpoint
     * Corner 5 6 7 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*5 - 3 + i];
	tri[1][i] = coor[3*6 - 3 + i];
	tri[2][i] = coor[3*7 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[11]=x_kvpt3s(pp,tri,&ier);
    if (istat[11] == 2) {
	return istat[11];
    }

    /*
     * --------------------------------------------------------------------------
     * Tetrahedron construction no. 12
     * There are four corners made up of edges and midpoint
     * Corner 6 7 8 M
     * --------------------------------------------------------------------------
     */
    for (i=0;i<3;i++) {
	tri[0][i] = coor[3*6 - 3 + i];
	tri[1][i] = coor[3*7 - 3 + i];
	tri[2][i] = coor[3*8 - 3 + i];
	tri[3][i] = pm[i];
    }
    istat[12]=x_kvpt3s(pp,tri,&ier);
    if (istat[12] == 2) {
	return istat[12];
    }


    isum=0;
    for (i=1;i<=12;i++) {
	isum+=istat[i];
    }
    /* think this is a quick one ... */
    if (isum >= 1) {
	return 2;
    }

    /* nothing found */
    return 0;
}



/*
 *=============================================================================
 * from ECLPOST kvpt3s.f
 *=============================================================================
 */

int x_kvpt3s (
	      double pp[],
	      double tri[][3],
	      int   *ier
	      )
{

    double czero, cone;
    double amat[3][3], bv[3], xv[3], eps, xdum;
    int   ierl;
    int   i, ipiv[3];

    /* initilise */
    ierl=0;

    czero=0.0e0;
    cone=1.0e0;

    /* transforming */

    for (i=0; i<3; i++) {
	amat[i][0] = tri[1][i]-tri[0][i];
	amat[i][1] = tri[2][i]-tri[0][i];
	amat[i][2] = tri[3][i]-tri[0][i];
	bv[i]      = pp[i] -tri[0][i];

    }





    eps=1.0e-05;
    xdum=0.0;

    x_kmgmps(amat,ipiv,xdum,3,3,eps,&ierl);


    if (ierl == -2) return -2;
    if (ierl !=  0) return -9;


    x_kmsubs(xv,amat,3,3,bv,ipiv,&ierl);


    if (ierl != 0) return -5;

    /* check if point is inside unit tetrehedron */

    /* outside */
    if (xv[0] < czero) return 0;
    if (xv[1] < czero) return 0;
    if (xv[2] < czero) return 0;
    if ((xv[0]+xv[1]+xv[2]) > cone) return 0;

    /* on edge */
    if (xv[0] == 0.0) return 1;
    if (xv[1] == 0.0) return 1;
    if (xv[2] == 0.0) return 1;
    if ((xv[0]+xv[1]+xv[2]) == cone) {
	return 1;
    }
    else{
	return 2;
    }
}

void x_kmgmps (
	       double a[][3],
	       int    l[],
	       double prmn,
	       int    m,
	       int    n,
	       double eps,
	       int    *ier
	       )
{
    int n1, i, ip, iq, j, k, k1;
    double am, amx, pr, cnull, cen;

    cnull=0.0e0;
    cen  =1.0e0;


    /* error check */

    /* om n != m - skippes */

    /* init: */
    *ier=0;
    n1=n-1;
    amx=cnull;
    prmn=cen;


    /* OOPS fabs for floats/doubles, not abs!! */

    for (i=0;i<n;i++) {
	l[i]=i;
	for (j=0;j<n;j++) {
	    if (amx < fabs(a[i][j])) amx=fabs(a[i][j]);
	}
    }
    if (amx <= cnull ) {
	*ier=-2;
	return;
    }

    /* elimination */
    for (k=0;k<n1;k++){
	am=cnull;
	for (i=k;i<n;i++){
	    if (fabs(a[l[i]][k]) > am) {
		am=fabs(a[l[i]][k]);
		j=i;
	    }
	}
	/* pivot element is a[l[j]][k] */

	/* singularity control */
	pr=am/amx;
	if (prmn >= pr) prmn=pr;

	if (pr <= eps) {
	    *ier=-2;
	    return;
	}


	/* interchange */
	ip=l[j];
	l[j]=l[k];
	l[k]=ip;

	k1=k+1;

	for (i=k1;i<n;i++) {
	    j=l[i];
	    am=a[j][k]/a[ip][k];
	    a[j][k]=am;
	    for (iq=k1;iq<n;iq++) {
		a[j][iq]=a[j][iq] - am*a[ip][iq];
	    }
	}
    }
}

/*
 *=============================================================================
 * Solve system Ax=b by substitution
 *=============================================================================
 */

void x_kmsubs (
	       double x[],
	       double a[][3],
	       int    m,
	       int    n,
	       double b[],
	       int    l[],
	       int    *ier
	       )
{
    int     n1, k, k1, i, j;
    double  s;

    /* init: */
    *ier=0;
    n1=n-1;

    /* elimination */


    for (k=0; k<n1; k++) {
	k1 = k+1;
	for (i=k1; i<n; i++) {
	    b[l[i]]=b[l[i]] - a[l[i]][k]*b[l[k]];
	}
    }

    /* set in: */

    x[n-1] = b[l[n-1]]/a[l[n-1]][n-1];


    for (k=(n1-1); k>=0; k--) {

	k1=k+1;
	i=l[k];
	s=b[i];
	for (j=k1; j<n; j++) {
	    s = s - a[i][j]*x[j];
	}
	x[k]=s/a[i][k];
    }
}
