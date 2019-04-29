/*
 *******************************************************************************
 *
 * NAME:
 *    map_operation value.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Basic scalar operation for map. The operation depends on mode value:
 *    -1    Set all map nodes UNDEF
 *     0    Set all map nodes to <value>
 *     1    Add <value> to all exisitng defined nodes
 *     2    Subtract <value> to all exisitng defined nodes
 *     3    Multiply <value> to all exisitng defined nodes
 *     4    Divide <value> to all existing defined nodes (value must not be 0)
 *     5    Set all UNDEF to value
 *
 *     7    if (z<value) then z=value2 else z=value3 (if value3 is not UNDEF)
 *     8    if (z<=value) then z=value2 else z=value3 (if value3 is not UNDEF)
 *     9    if (z>value) then z=value2 else z=value3 (if value3 is not UNDEF)
 *     10   if (z>=value) then z=value2 else z=value3 (if value3 is not UNDEF)
 *     11   if (z==value) then z=value2 else z=value3 (if value3 is not UNDEF)
 *
 *
 *
 * ARGUMENTS:
 *    mode           i     Kind of operation (see DESCRIPTION)
 *    nx, ny         i     Map dimension
 *    p_zval_v      i/o    Maps depth/whatever values vector (pointer)
 *    value          i     Value1
 *    value2         i     Value2
 *    value3         i     Value3
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void + Changed pointer to map property (z values)
 *
 * TODO/ISSUES/BUGS:
 *
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"


void map_operation_value (
			  int mode,
			  int nx,
			  int ny,
			  double *p_zval_v,
			  double value,
			  double value2,
			  double value3,
			  int debug
			  )
{
    int i, j, ib;
    char s[24]="map_operation_value";

    xtgverbose(debug);

    xtg_speak(s,2,"Map operation value: Using mode %3d",mode);

    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {

	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);


	    if (p_zval_v[ib]>UNDEF_MAP_LIMIT && mode==5) {
		p_zval_v[ib]=value;
	    }
	    else if (p_zval_v[ib]<UNDEF_MAP_LIMIT) {

		if (mode==-1) {                         /* all UNDEF  */
		    p_zval_v[ib]=UNDEF;
		}
		else if (mode==0) {                     /* 0 = set  */
		    p_zval_v[ib]=value;
		}
		else if (mode==1) {                     /* 1 = add  */
		    p_zval_v[ib]=p_zval_v[ib]+value;
		}
		else if (mode==2) {                     /* 2 = subtract  */
		    p_zval_v[ib]=p_zval_v[ib]-value;
		}
		else if (mode==3) {                     /* 3 = multiply  */
		    p_zval_v[ib]=p_zval_v[ib]*value;
		}
		else if (mode==4) {                     /* 4 = divide  */
		    p_zval_v[ib]=p_zval_v[ib]/value;
		}


		/* if map less than value then map = value2 (else value3) */
		/* modes <,<=,>,>=,== (7,8,9,10,11) */

		else if (mode==7) {
		    if (p_zval_v[ib]<value) {
			p_zval_v[ib]=value2;
		    }
		    else {
			/* "else" will be activated if value3 is not
			   set to UNDEF */

			if (value3 < UNDEF_LIMIT) {
			    p_zval_v[ib]=value3;
			}
		    }
		}
		/* if map <= than value then map = value2 (else value3) */
		else if (mode==8) {
		    if (p_zval_v[ib]<=value) {
			p_zval_v[ib]=value2;
		    }
		    else {
			/* "else" activated if value3 is not set to UNDEF */
			if (value3 < UNDEF_LIMIT) {
			    p_zval_v[ib]=value3;
			}
		    }
		}
		/* if map > than value then map = value2 (else value3) */
		else if (mode==9) {
		    if (p_zval_v[ib]>value) {
			p_zval_v[ib]=value2;
		    }
		    else {
			/* "else" activated if value3 is not set to UNDEF */
			if (value3 < UNDEF_LIMIT) {
			    p_zval_v[ib]=value3;
			}
		    }
		}
		/* if map >= than value then map = value2 (else value3) */
		else if (mode==10) {
		    if (p_zval_v[ib]>=value) {
			p_zval_v[ib]=value2;
		    }
		    else {
			/* "else" be activated if value3 is not set to UNDEF */
			if (value3 < UNDEF_LIMIT) {
			    p_zval_v[ib]=value3;
			}
		    }
		}
		/* if map == than value then map = value2 (else value3) */
		else if (mode==11) {
		    if (p_zval_v[ib]==value) {
			p_zval_v[ib]=value2;
		    }
		    else {
			/* "else" activated if value3 is not set to UNDEF */
			if (value3 < UNDEF_LIMIT) {
			    p_zval_v[ib]=value3;
			}
		    }
		}
	    }
	}
    }
}
