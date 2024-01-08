
import cupy
from cupyx.scipy.spatial.delaunay_2d._schewchuk import SCHEWCHUK_DEF


KERNEL_DIVISION = SCHEWCHUK_DEF + r"""

#define INLINE_H_D __forceinline__ __device__
#define DIM     2
#define DEG     ( DIM + 1 )

using RealType = double;

#define BLOCK_DIM    blockDim.x
#define THREAD_IDX   threadIdx.x

__forceinline__ __device__ int getCurThreadIdx()
{
    const int threadsPerBlock   = blockDim.x;
    const int curThreadIdx    = ( blockIdx.x * threadsPerBlock ) + threadIdx.x;
    return curThreadIdx;
}

__forceinline__ __device__ int getThreadNum()
{
    const int blocksPerGrid     = gridDim.x;
    const int threadsPerBlock   = blockDim.x;
    const int threadNum         = blocksPerGrid * threadsPerBlock;
    return threadNum;
}

enum Counter {
    CounterExact,
    CounterFlag,
    CounterNum
};

///////////////////////////////////////////////////////////////////// Orient //

enum Orient
{
    OrientNeg   = -1,
    OrientZero  = +0,
    OrientPos   = +1
};

INLINE_H_D Orient flipOrient( Orient ord )
{
    return ( OrientPos == ord ) ? OrientNeg : OrientPos;
}

// Our 2D orientation is the same as Shewchuk
INLINE_H_D Orient ortToOrient( RealType det )
{
    return ( det > 0 ) ? OrientPos : ( ( det < 0 ) ? OrientNeg : OrientZero );
}

struct Point2
{
    RealType _p[ 2 ];

    INLINE_H_D bool lessThan( const Point2& pt ) const
    {
        if ( _p[0] < pt._p[0] ) return true;
        if ( _p[0] > pt._p[0] ) return false;
        if ( _p[1] < pt._p[1] ) return true;

        return false;
    }

    INLINE_H_D bool operator < ( const Point2& pt ) const
    {
        return lessThan( pt );
    }

    INLINE_H_D bool equals( const Point2& pt ) const
    {
        return ( _p[0] == pt._p[0] && _p[1] == pt._p[1] );
    }

    INLINE_H_D bool operator == ( const Point2& pt ) const
    {
        return equals( pt );
    }

    INLINE_H_D bool operator != ( const Point2& pt ) const
    {
        return !equals( pt );
    }
};

struct Tri
{
    int _v[3];

    INLINE_H_D bool has( int v ) const
    {
        return ( _v[0] == v || _v[1] == v || _v[2] == v );
    }

    INLINE_H_D int getIndexOf( int v ) const
    {
        if ( _v[0] == v ) return 0;
        if ( _v[1] == v ) return 1;
        if ( _v[2] == v ) return 2;

        return -1;
    }

    INLINE_H_D bool equals( const Tri& tri ) const
    {
        return ( _v[0] == tri._v[0] && _v[1] == tri._v[1] &&
                 _v[2] == tri._v[2] );
    }

    INLINE_H_D bool operator == ( const Tri& pt ) const
    {
        return equals( pt );
    }

    INLINE_H_D bool operator != ( const Tri& pt ) const
    {
        return !equals( pt );
    }
};

INLINE_H_D Tri makeTri( int v0, int v1, int v2 )
{
    const Tri newTri = { v0, v1, v2 };

    return newTri;
}

INLINE_H_D bool isValidTriVi( int vi )
{
    return ( vi >= 0 && vi < DEG );
}

// ...76543210
//        ^^^^ vi (2 bits)
//        ||__ constraint
//        |___ special
// Rest is triIdx

template< typename T >
INLINE_H_D bool isBitSet( T c, int bit )
{
    return ( 1 == ( ( c >> bit ) & 1 ) );
}

template< typename T >
INLINE_H_D void setBitState( T& c, int bit, bool state )
{
    const T val = ( 1 << bit );
    c = state
        ? ( c | val )
        : ( c & ~val );
}

// Get the opp tri and vi
// Retain some states: constraint
// Clear some states:  special
INLINE_H_D int getOppValTriVi( int val )
{
    return ( val & ~0x08 );
}

INLINE_H_D int getOppValTri( int val )
{
    return (val >> 4);
}

INLINE_H_D int getOppValVi( int val )
{
    return (val & 3);
}

INLINE_H_D bool isOppValConstraint( int val )
{
    return isBitSet( val, 2 );
}

INLINE_H_D void setOppValTri( int &val, int idx )
{
    val = (val & 0x0F) | (idx << 4);
}

// Set the opp tri and vi
// Retain some states: constraint
// Clear some states:  special
INLINE_H_D void setOppValTriVi( int &val, int idx, int vi )
{
    val = (idx << 4) | (val & 0x04) | vi;
}

INLINE_H_D int getTriIdx( int input, int oldVi )
{
    int idxVi = ( input >> (oldVi * 4) ) & 0xf;

    return ( idxVi >> 2 ) & 0x3;
}

INLINE_H_D int getTriVi( int input, int oldVi )
{
    int idxVi = ( input >> (oldVi * 4) ) & 0xf;

    return idxVi & 0x3;
}

struct TriOpp
{
    int _t[3];

    INLINE_H_D void setOpp( int vi, int triIdx, int oppTriVi )
    {
        _t[ vi ] = ( triIdx << 4 ) | oppTriVi;
    }

    INLINE_H_D void setOpp(
            int vi, int triIdx, int oppTriVi, bool isConstraint )
    {
        _t[ vi ] = ( triIdx << 4 ) | ( isConstraint << 2 ) | oppTriVi;
    }

    INLINE_H_D void setOppTriVi( int vi, int triIdx, int oppTriVi )
    {
        setOppValTriVi( _t[ vi ], triIdx, oppTriVi );
    }

    INLINE_H_D void setOppConstraint( int vi, bool val )
    {
        setBitState( _t[ vi ], 2, val );
    }

    INLINE_H_D void setOppSpecial( int vi, bool state )
    {
        setBitState( _t[ vi ], 3, state );
    }

    INLINE_H_D bool isNeighbor( int triIdx ) const
    {
        return ( (_t[0] >> 4) == triIdx ||
                 (_t[1] >> 4) == triIdx ||
                 (_t[2] >> 4) == triIdx );
    }

    INLINE_H_D int getIdxOf( int triIdx ) const
    {
        if ( ( _t[0] >> 4 ) == triIdx ) return 0;
        if ( ( _t[1] >> 4 ) == triIdx ) return 1;
        if ( ( _t[2] >> 4 ) == triIdx ) return 2;
        return -1;
    }

    INLINE_H_D bool isOppSpecial( int vi ) const
    {
        return isBitSet( _t[ vi ], 3 );
    }

    INLINE_H_D int getOppTriVi( int vi ) const
    {
        if ( -1 == _t[ vi ] )
            return -1;

        return getOppValTriVi( _t[ vi ] );
    }

    INLINE_H_D bool isOppConstraint( int vi ) const
    {
        return isOppValConstraint( _t[ vi ] );
    }

    INLINE_H_D int getOppTri( int vi ) const
    {
        return getOppValTri( _t[ vi ] );
    }

    INLINE_H_D void setOppTri( int vi, int idx )
    {
        return setOppValTri( _t[ vi ], idx );
    }

    INLINE_H_D int getOppVi( int vi ) const
    {
        return getOppValVi( _t[ vi ] );
    }
};

template< typename T >
__forceinline__ __device__ void cuSwap( T& v0, T& v1 )
{
    const T tmp = v0;
    v0          = v1;
    v1          = tmp;

    return;
}

template< typename T>
__forceinline__ __device__ void storeIntoBuffer(
        T* s_buffer, int* s_num, const T& item )
{
    const int idx = atomicAdd( s_num, 1 );

    s_buffer[ idx ] = item;
}

__global__ void kerMakeFirstTri
(
Tri*	triArr,
TriOpp*	oppArr,
char*	triInfoArr,
Tri*     initTri,
int     infIdx
)
{
    Tri tri = initTri[0];

    const Tri tris[] = {
        { tri._v[0], tri._v[1], tri._v[2] },
        { tri._v[2], tri._v[1], infIdx },
        { tri._v[0], tri._v[2], infIdx },
        { tri._v[1], tri._v[0], infIdx }
    };

    const int oppTri[][3] = {
        { 1, 2, 3 },
        { 3, 2, 0 },
        { 1, 3, 0 },
        { 2, 1, 0 }
    };

    const int oppVi[][4] = {
        { 2, 2, 2 },
        { 1, 0, 0 },
        { 1, 0, 1 },
        { 1, 0, 2 }
    };

    for ( int i = 0; i < 4; ++i )
    {
        triArr[ i ]     = tris[ i ];
        triInfoArr[ i ] = 1;

        TriOpp opp = { -1, -1, -1 };

        for ( int j = 0; j < 3; ++j )
            opp.setOpp( j, oppTri[i][j], oppVi[i][j] );

        oppArr[ i ] = opp;
    }
}

__device__ const int SplitFaces[6][3] = {
    /*0*/ { 0, 3 },
    /*1*/ { 2, 3 },                   /*2*/ { 1, 3 },
    /*3*/ { 1, 2 },  /*4*/ { 2, 0 },  /*5*/ { 0, 1 }
};

__device__ const int SplitNext[6][2] = {
    { 1, 2 },
    { 3, 4 },               { 5, 3 },
    { 1, 0 },   { 2, 0 },   { 3, 0 },
};

__forceinline__ __device__ const Point2& getPoint(
        int idx, Point2* pointArr )
{
    return pointArr[ idx ];
}

__forceinline__ __device__ Orient doOrient2DFastExact
(
const RealType* p0,
const RealType* p1,
const RealType* p2,
RealType*  predConsts
)
{
    const RealType det = orient2dFastExact( predConsts, p0, p1, p2 );
    return ortToOrient( det );
}

__forceinline__ __device__ Orient doOrient2DFast(
        int v0, int v1, int v2, int infIdx,
        Point2* points, RealType*  predConsts )
{
    const RealType* pt[] = {
        getPoint(v0, points)._p,
        getPoint(v1, points)._p,
        getPoint(v2, points)._p };

    RealType det = orient2dFast( predConsts, pt[0], pt[1], pt[2] );

    if ( v0 == infIdx | v1 == infIdx | v2 == infIdx )
        det = -det;

    return ortToOrient( det );
}

__forceinline__ __device__ Orient doOrient2DSoSOnly
(
const RealType* p0,
const RealType* p1,
const RealType* p2,
int v0,
int v1,
int v2
)
{
    ////
    // Sort points using vertex as key, also note their sorted order
    ////
    const RealType* p[DEG] = { p0, p1, p2 };
    int pn = 1;

    if ( v0 > v1 ) { cuSwap( v0, v1 ); cuSwap( p[0], p[1] ); pn = -pn; }
    if ( v0 > v2 ) { cuSwap( v0, v2 ); cuSwap( p[0], p[2] ); pn = -pn; }
    if ( v1 > v2 ) { cuSwap( v1, v2 ); cuSwap( p[1], p[2] ); pn = -pn; }

    RealType result = 0;
    int depth;

    for ( depth = 1; depth <= 4; ++depth )
    {
        switch ( depth )
        {
        case 1:
            result = p[2][0] - p[1][0];
            break;
        case 2:
            result = p[1][1] - p[2][1];
            break;
        case 3:
            result = p[0][0] - p[2][0];
            break;
        default:
            result = 1.0;
            break;
        }

        if ( result != 0 )
            break;
    }

    const RealType det = result * pn;

    return ortToOrient( det );
}

__forceinline__ __device__ Orient doOrient2DFastExactSoS(
int        v0,
int        v1,
int        v2,
int        infIdx,
Point2*    points,
int*       orgPointIdx,
RealType*  predConsts )
{
    const RealType* pt[] = {
        getPoint(v0, points)._p,
        getPoint(v1, points)._p,
        getPoint(v2, points)._p };

    // Fast-Exact
    Orient ord = doOrient2DFastExact( pt[0], pt[1], pt[2], predConsts );

    if ( OrientZero == ord )
    {
        // SoS
        if ( orgPointIdx != NULL )
        {
            v0 = orgPointIdx[ v0 ];
            v1 = orgPointIdx[ v1 ];
            v2 = orgPointIdx[ v2 ];
        }

        ord = doOrient2DSoSOnly( pt[0], pt[1], pt[2], v0, v1, v2 );
    }

    if ( (v0 == infIdx) | (v1 == infIdx) | (v2 == infIdx) )
        ord = flipOrient( ord );

    return ord;
}

template<bool doFast>
__forceinline__ __device__ int initPointLocation(
int         idx,
Tri         tri,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts )
{
    if ( tri.has( idx ) || idx == infIdx ) return -1;   // Already inserted

    const int triVert[5] = { tri._v[0], tri._v[1], tri._v[2], infIdx };
    int face = 0;

    for ( int i = 0; i < 3; ++i )
    {
        const int *fv = SplitFaces[ face ];

        Orient ort = ( doFast )
            ? doOrient2DFast( triVert[ fv[0] ], triVert[ fv[1] ], idx,
                              infIdx, points, predConsts )
            : doOrient2DFastExactSoS( triVert[ fv[0] ], triVert[ fv[1] ], idx,
                                      infIdx, points, orgPointIdx,
                                      predConsts );

        // Needs exact computation
        if ( doFast && (ort == OrientZero) ) return -2;

        // Use the reverse direction 'cause the splitting point is Infty!
        face = SplitNext[ face ][ ( ort == OrientPos ) ? 1 : 0 ];
    }

    return face;
}

__global__ void kerInitPointLocationFast
(
int*        vertTriVec,
int         nVert,
int*        exactCheckArr,
int*        counter,
Tri*         tri,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts
)
{
    // Iterate points
    for ( int idx = getCurThreadIdx(); idx < nVert; idx += getThreadNum() )
    {
        const int loc = initPointLocation<true>(
            idx, tri[0], infIdx, points, orgPointIdx, predConsts );

        if ( loc != -2 )
            vertTriVec[ idx ] = loc;
        else
            storeIntoBuffer( exactCheckArr, &counter[ CounterExact ], idx );
    }
}

__global__ void kerInitPointLocationExact
(
int*    vertTriArr,
int         nVert,
int*    exactCheckArr,
int*    counter,
Tri*     tri,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts
)
{
    const int exactNum = counter[ CounterExact ];

    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        const int vertIdx = exactCheckArr[ idx ];

        vertTriArr[ vertIdx ] = initPointLocation<false>(
            vertIdx, tri[0], infIdx, points, orgPointIdx, predConsts );
    }
}
"""

DELAUNAY_MODULE = cupy.RawModule(
    code=KERNEL_DIVISION, options=('-std=c++11',),
    name_expressions=['kerMakeFirstTri', 'kerInitPointLocationFast',
                      'kerInitPointLocationExact'])


N_BLOCKS = 512
BLOCK_SZ = 128


def make_first_tri(tri_arr, opp_arr, tri_info_arr, tri, inf_idx):
    ker_make_first_tri = DELAUNAY_MODULE.get_function('kerMakeFirstTri')
    ker_make_first_tri(
        (1,), (1,), (tri_arr, opp_arr, tri_info_arr, tri, inf_idx))


def init_point_location_fast(
        vert_tri, n_vert, exact_check, counter, tri, inf_idx,
        point_vec, points_idx, pred_consts):
    ker_init_point_loc_fast = DELAUNAY_MODULE.get_function(
        'kerInitPointLocationFast')
    ker_init_point_loc_fast((N_BLOCKS,), (BLOCK_SZ,), (
        vert_tri, n_vert, exact_check, counter, tri, inf_idx,
        point_vec, points_idx, pred_consts))


def init_point_location_exact(
        vert_tri, n_vert, exact_check, counter, tri, inf_idx,
        point_vec, points_idx, pred_consts):
    ker_init_point_loc_fast = DELAUNAY_MODULE.get_function(
        'kerInitPointLocationExact')
    ker_init_point_loc_fast((N_BLOCKS,), (BLOCK_SZ,), (
        vert_tri, n_vert, exact_check, counter, tri, inf_idx,
        point_vec, points_idx, pred_consts))
