/*
 * Setting (and returning) verbose level
 */

int xtgverbose(int iv) {

    static int iverb;

    if (iv > -1) {
	iverb=iv;
    }

    return iverb;
}
