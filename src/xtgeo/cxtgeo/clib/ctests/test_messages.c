/* Testing the XTG message system for C side
 * JRIV
 */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>

int main () {
    char s[24]="myroutine";

    plan(NO_PLAN);
    ok(3 == 3);

    xtgverbose(1);
    xtg_verbose_file("NONE");
    /*
     * -------------------------------------------------------------------------
     * XTG_speak
     */
    xtg_speak(s,1,"Hello world for %s",s);
    

    done_testing();

}
