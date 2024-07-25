
import cupy
from cupyx.scipy.spatial.delaunay_2d._schewchuk import SCHEWCHUK_DEF


KERNEL_DIVISION = SCHEWCHUK_DEF + r"""

#define INLINE_H_D __forceinline__ __device__
#define DIM     2
#define DEG     ( DIM + 1 )

using RealType = double;

#define BLOCK_DIM    blockDim.x
#define THREAD_IDX   threadIdx.x

#define INT_MAX   0x7FFFFFFF

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

INLINE_H_D void setTriIdxVi(
int &output,
int oldVi,
int ni,
int newVi
)
{
    output -= ( 0xF ) << ( oldVi * 4 );
    output += ( ( ni << 2) + newVi ) << ( oldVi * 4 );
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

    INLINE_H_D Point2 operator - ( const Point2& pt ) const
    {
        return Point2{{ _p[0] - pt._p[0], _p[1] -  pt._p[1]}};
    }

    INLINE_H_D Point2 operator + ( const Point2& pt ) const
    {
        return Point2{{ _p[0] + pt._p[0], _p[1] + pt._p[1]}};
    }

    INLINE_H_D Point2 operator * ( const double v ) const
    {
        return Point2{{ _p[0] * v, _p[1] * v}};
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

// Tri info
// 76543210
//     ^^^^ 0: Dead      1: Alive
//     |||_ 0: Checked   1: Changed
//     ||__ PairType

enum TriCheckState
{
    Checked,
    Changed,
};

enum PairType
{
    PairNone    = 0,
    PairSingle  = 1,
    PairDouble  = 2,
    PairConcave = 3
};

INLINE_H_D bool isTriAlive( char c )
{
    return isBitSet( c, 0 );
}

INLINE_H_D void setTriAliveState( char& c, bool b )
{
    setBitState( c, 0, b );
}

INLINE_H_D TriCheckState getTriCheckState( char c )
{
    return isBitSet( c, 1 ) ? Changed : Checked;
}

INLINE_H_D void setTriCheckState( char& c, TriCheckState s )
{
    if ( Checked == s ) setBitState( c, 1, false );
    else                setBitState( c, 1, true );
}

INLINE_H_D void setTriPairType( char& c, PairType p )
{
    c = ( c & 0xF3 ) | ( p << 2 );
}

INLINE_H_D PairType getTriPairType( char c )
{
    return (PairType) (( c >> 2 ) & 3);
}

enum CheckDelaunayMode
{
    CircleFastOrientFast,
    CircleExactOrientSoS
};

enum ActTriMode
{
    ActTriMarkCompact,
    ActTriCollectCompact
};

/////////////////////////////////////////////////////////////////////// Side //

enum Side
{
    SideIn   = -1,
    SideZero = +0,
    SideOut  = +1
};

INLINE_H_D Side cicToSide( RealType det )
{
    return ( det < 0 ) ? SideOut : ( ( det > 0 ) ? SideIn : SideZero );
}

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

__device__ RealType orient2dDet
(
const RealType* predConsts,
const RealType *pa,
const RealType *pb,
const RealType *pc
)
{
    RealType detleft, detright;

    detleft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    detright = (pa[1] - pc[1]) * (pb[0] - pc[0]);

    return detleft - detright;
}

__forceinline__ __device__ RealType incircleDet
(
const RealType* predConsts,
const RealType* pa,
const RealType* pb,
const RealType* pc,
const RealType* pd
)
{
    RealType adx, bdx, cdx, ady, bdy, cdy;
    RealType bdxcdy, cdxbdy, cdxady, adxcdy, adxbdy, bdxady;
    RealType alift, blift, clift;
    RealType det;

    adx = pa[0] - pd[0];
    bdx = pb[0] - pd[0];
    cdx = pc[0] - pd[0];
    ady = pa[1] - pd[1];
    bdy = pb[1] - pd[1];
    cdy = pc[1] - pd[1];

    bdxcdy = bdx * cdy;
    cdxbdy = cdx * bdy;
    alift = adx * adx + ady * ady;

    cdxady = cdx * ady;
    adxcdy = adx * cdy;
    blift = bdx * bdx + bdy * bdy;

    adxbdy = adx * bdy;
    bdxady = bdx * ady;
    clift = cdx * cdx + cdy * cdy;

    det = alift * (bdxcdy - cdxbdy)
        + blift * (cdxady - adxcdy)
        + clift * (adxbdy - bdxady);

    return det;
}

__forceinline__ __device__ float inCircleDet
(
Tri tri,
int vert,
/** predWrapper values **/
int         infIdx,
Point2*     points,
RealType*   predConsts
)
{
    const RealType* pt[] = {
        getPoint( tri._v[0], points )._p,
        getPoint( tri._v[1], points )._p,
        getPoint( tri._v[2], points )._p,
        getPoint( vert, points )._p };

    float det;

    if ( tri.has( infIdx ) )
    {
        const int infVi = tri.getIndexOf( infIdx );

        det = orient2dDet(
            predConsts, pt[ (infVi + 1) % 3 ], pt[ (infVi + 2) % 3 ], pt[3] );
    }
    else
        det = incircleDet( predConsts, pt[0], pt[1], pt[2], pt[3] );

    return det;
}


__global__ void
kerVoteForPoint
(
int*        vertexTriVec,
int         nVert,
Tri*        triArr,
int*        vertCircleArr,
int*        triCircleArr,
int         noSample,
/** predWrapper values **/
int         infIdx,
Point2*     points,
RealType*   predConsts
)
{
    const float rate = float(nVert) / noSample;

    // Iterate uninserted points
    for ( int idx = getCurThreadIdx(); idx < noSample; idx += getThreadNum() )
    {
        const int vert = int( idx * rate );

        //*** Compute insphere value
        const int triIdx   = vertexTriVec[ vert ];

        if ( -1 == triIdx ) continue;

        const Tri tri = triArr[ triIdx ];
        float cval    = /*hash( idx );*/ inCircleDet(
            tri, vert, infIdx, points, predConsts );

        //*** Sanitize and store sphere value

        if ( cval <= 0 )
            cval = 0;

        int ival = __float_as_int(cval);

        vertCircleArr[ idx ] =  ival;

        //*** Vote
        if ( triCircleArr[ triIdx ] < ival )
            atomicMax( &triCircleArr[ triIdx ], ival );
    }

    return;
}

__global__ void
kerPickWinnerPoint
(
int*         vertexTriVec,
int          nVert,
int*         vertCircleArr,
int*         triCircleArr,
int*         triVertArr,
int          noSample
)
{
    const float rate = float(nVert) / noSample;

    // Iterate uninserted points
    for ( int idx = getCurThreadIdx(); idx < noSample; idx += getThreadNum() )
    {
        const int vert   = int( idx * rate );
        const int triIdx = vertexTriVec[ vert ];

        if ( triIdx == -1 ) continue;

        const int vertSVal = vertCircleArr[ idx ];
        const int winSVal  = triCircleArr[ triIdx ];

        // Check if vertex is winner

        if ( winSVal == vertSVal )
            atomicMin( &triVertArr[ triIdx ], vert );
    }

    return;
}

template< typename T >
__global__ void
kerShiftValues
(
int*        shiftVec,
int         nShift,
T*          src,
T*          dest
)
{
    for ( int idx = getCurThreadIdx(); idx < nShift; idx += getThreadNum() )
    {
        const int shift = shiftVec[ idx ];
        dest[ idx + shift ] = src[ idx ];
    }
}

__global__ void
kerShiftOpp
(
int*        shiftVec,
int         nShift,
TriOpp*     src,
TriOpp*     dest
)
{
    for ( int idx = getCurThreadIdx(); idx < nShift; idx += getThreadNum() )
    {
        const int shift = shiftVec[ idx ];

        TriOpp opp = src[ idx ];

        for ( int vi = 0; vi < 3; ++vi )
        {
            const int oppTri = opp.getOppTri( vi );
            opp.setOppTri( vi, oppTri + shiftVec[ oppTri ] );
        }

        // CudaAssert( idx + shift < destSize );

        dest[ idx + shift ] = opp;
    }
}

__global__ void
kerShiftTriIdx
(
int*        idxVec,
int         nIdx,
int*        shiftArr
)
{
    for ( int idx = getCurThreadIdx(); idx < nIdx; idx += getThreadNum() )
    {
        const int oldIdx = idxVec[ idx ];

        if ( oldIdx != -1 )
            idxVec[ idx ] = oldIdx + shiftArr[ oldIdx ];
    }
}

template < bool doFast >
__forceinline__ __device__ bool splitPoints
(
int     vertex,
int&    vertTriIdx,
int*    triToVert,
Tri*    triArr,
int*    insTriMap,
int     triNum,
int     insTriNum,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts
)
{
    int triIdx = vertTriIdx;

    if ( triIdx == -1 ) return true;   // Point inserted

    const int splitVertex = triToVert[ triIdx ];

    // Vertex's triangle will not be split in this round
    if ( splitVertex >= INT_MAX - 1 ) return true;

    // Fast mode, *this* vertex will split its triangle
    if ( doFast && vertex == splitVertex )
    {
        vertTriIdx = -1;
        return true;
    }

    const Tri tri         = triArr[ triIdx ];
    const int newBeg      = ( triNum >= 0 ) ? (
        triNum + 2 * insTriMap[ triIdx ] ) : ( triIdx + 1 );
    const int triVert[4]  = { tri._v[0], tri._v[1], tri._v[2], splitVertex };

    int face = 0;

    for ( int i = 0; i < 2; ++i )
    {
        const int *fv = SplitFaces[ face ];

        Orient ort = ( doFast )
            ? doOrient2DFast(
                triVert[ fv[0] ], triVert[ fv[1] ], vertex, infIdx, points,
                predConsts )
            : doOrient2DFastExactSoS(
                triVert[ fv[0] ], triVert[ fv[1] ], vertex, infIdx, points,
                orgPointIdx, predConsts );

        // Needs exact computation
        if ( doFast && (ort == OrientZero) ) return false;

        face = SplitNext[ face ][ ( ort == OrientPos ) ? 0 : 1 ];
    }

    vertTriIdx = ( ( face == 3 ) ? triIdx : (newBeg + face - 4) );

    return true;
}

__global__ void
kerSplitPointsFast
(
int*        vertTriVec,
int         vertTriNum,
int*        triToVert,
Tri*        triArr,
int*        insTriMap,
int*        exactCheckArr,
int*        counter,
int         triNum,
int         insTriNum,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts
)
{
    // Iterate points
    for ( int idx = getCurThreadIdx(); idx < vertTriNum; idx += getThreadNum())
    {
        bool ret = splitPoints<true>( idx, vertTriVec[ idx ], triToVert,
            triArr, insTriMap, triNum, insTriNum, infIdx, points, orgPointIdx,
            predConsts);

        if ( !ret )
            storeIntoBuffer( exactCheckArr, &counter[ CounterExact ], idx );
    }
}

__global__ void
kerSplitPointsExactSoS
(
int*    vertTriArr,
int*    triToVert,
Tri*    triArr,
int*    insTriMap,
int*    exactCheckArr,
int*    counter,
int     triNum,
int     insTriNum,
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

        splitPoints< false >( vertIdx, vertTriArr[ vertIdx ], triToVert,
            triArr, insTriMap, triNum, insTriNum, infIdx, points, orgPointIdx,
            predConsts );
    }
}

__global__ void
kerSplitTri
(
int*        splitTriArr,
int         nSplitTri,
Tri*        triArr,
TriOpp*     oppArr,
char*       triInfoArr,
int*        insTriMap,
int*        triToVert,
int         triNum,
int         insTriNum
)
{
    // Iterate current triangles
    for ( int idx = getCurThreadIdx(); idx < nSplitTri; idx += getThreadNum() )
    {
        const int triIdx         = splitTriArr[ idx ];
        const int newBeg         = ( triNum >= 0 ) ? (
            triNum + 2 * insTriMap[ triIdx ] ) : ( triIdx + 1 );
        const int newTriIdx[DEG] = { triIdx, newBeg, newBeg + 1 };
        TriOpp newOpp[3]         = {
            { -1, -1, -1 },
            { -1, -1, -1 },
            { -1, -1, -1 }
        };

        // Set adjacency of 3 internal faces of 3 new triangles
        newOpp[ 0 ].setOpp( 0, newTriIdx[ 1 ], 1 );
        newOpp[ 0 ].setOpp( 1, newTriIdx[ 2 ], 0 );
        newOpp[ 1 ].setOpp( 0, newTriIdx[ 2 ], 1 );
        newOpp[ 1 ].setOpp( 1, newTriIdx[ 0 ], 0 );
        newOpp[ 2 ].setOpp( 0, newTriIdx[ 0 ], 1 );
        newOpp[ 2 ].setOpp( 1, newTriIdx[ 1 ], 0 );

        // Set adjacency of 4 external faces
        const TriOpp oldOpp       = oppArr[ triIdx ];

        // Iterate faces of old triangle
        for ( int ni = 0; ni < DEG; ++ni )
        {
            if ( -1 == oldOpp._t[ ni ] ) continue; // No neighbour at this face

            int neiTriIdx = oldOpp.getOppTri( ni );
            int neiTriVi  = oldOpp.getOppVi( ni );

            // Check if neighbour has split
            const int neiNewBeg = insTriMap[ neiTriIdx ];

            if ( -1 == neiNewBeg ) // Neighbour is un-split
            {
                // Point un-split neighbour back to this new triangle
                oppArr[ neiTriIdx ].setOpp( neiTriVi, newTriIdx[ ni ], 2 );
            }
            else // Neighbour has split
            {
                // Get neighbour's new split triangle that has this face
                if ( triNum >= 0 )
                    neiTriIdx = (( 0 == neiTriVi ) ? neiTriIdx : (
                        triNum + 2 * neiNewBeg + neiTriVi - 1));
                else
                    neiTriIdx += neiTriVi;

                neiTriVi  = 2;
            }

            // Point this triangle to neighbour
            newOpp[ ni ].setOpp( 2, neiTriIdx, neiTriVi );
        }

        // Write split triangle and opp
        // Note: This slot will be overwritten below
        const Tri tri           = triArr[ triIdx ];
        const int splitVertex   = triToVert[ triIdx ];

        for ( int ti = 0; ti < DEG; ++ti )
        {
            const Tri newTri = {
                tri._v[ ( ti + 1 ) % DEG ],
                tri._v[ ( ti + 2 ) % DEG ],
                splitVertex
            };

            const int toTriIdx = newTriIdx[ ti ];
            triArr[ toTriIdx ] = newTri;
            oppArr[ toTriIdx ] = newOpp[ ti ];
            setTriAliveState( triInfoArr[ toTriIdx ], true );
            setTriCheckState( triInfoArr[ toTriIdx ], Changed );
        }
    }

    return;
}

__global__ void isTriActive(char* triInfo, bool* out, int nTriInfo) {
    for ( int idx = getCurThreadIdx(); idx < nTriInfo; idx += getThreadNum() )
    {
        out[idx] = ( isTriAlive( triInfo[idx] ) && (
            Changed == getTriCheckState( triInfo[idx] ) ) );
    }
}

__global__ void
kerMarkSpecialTris
(
char*        triInfoVec,
TriOpp*      oppArr,
int          nTriInfo
)
{
    for ( int idx = getCurThreadIdx(); idx < nTriInfo; idx += getThreadNum() )
    {
        if ( !isTriAlive( triInfoVec[ idx ] ) ) continue;

        TriOpp opp = oppArr[ idx ];

        bool changed = false;

        for ( int vi = 0; vi < DEG; ++vi )
        {
            if ( -1 == opp._t[ vi ] ) continue;

            if ( opp.isOppSpecial( vi ) )
                changed = true;
        }

        if ( changed )
            setTriCheckState( triInfoVec[ idx ], Changed );
    }
}

__forceinline__ __device__ int encode( int triIdx, int vi )
{
    return ( triIdx << 2 ) | vi;
}

__forceinline__ __device__ void decode( int code, int* idx, int* vi )
{
    *idx = ( code >> 2 );
    *vi = ( code & 3 );
}

__forceinline__ __device__ void
voteForFlip
(
int* triVoteArr,
int  botTi,
int  topTi,
int  botVi
)
{
    const int voteVal = encode( botTi, botVi );

    atomicMin( &triVoteArr[ botTi ],  voteVal );
    atomicMin( &triVoteArr[ topTi ],  voteVal );
}

/////////////////////////////////////////////////////////////////// InCircle //

__forceinline__ __device__ Side doInCircleFast(
Tri         tri,
int         vert,
/** predWrapper values **/
int         infIdx,
Point2*     points,
RealType*   predConsts
)
{
    const RealType* pt[] = {
        getPoint( tri._v[0], points )._p,
        getPoint( tri._v[1], points )._p,
        getPoint( tri._v[2], points )._p,
        getPoint( vert, points )._p };

    if ( vert == infIdx )
        return SideOut;

    RealType det;

    if ( tri.has( infIdx ) )
    {
        const int infVi = tri.getIndexOf( infIdx );

        det = orient2dFast(
            predConsts, pt[ (infVi + 1) % 3 ], pt[ (infVi + 2) % 3 ], pt[3] );
    }
    else
        det = incircleFast( predConsts, pt[0], pt[1], pt[2], pt[3] );

    return cicToSide( det );
}

__forceinline__ __device__ Side doInCircleFastExact(
    const RealType* p0, const RealType* p1, const RealType* p2,
    const RealType* p3, RealType*   predConsts )
{
    RealType det = incircleFastAdaptExact( predConsts, p0, p1, p2, p3 );

    return cicToSide( det );
}

__forceinline__ __device__ RealType doOrient1DExact_Lifted
(
const RealType* p0,
const RealType* p1,
RealType*       predConsts
)
{
    const RealType det = orient1dExact_Lifted( predConsts, p0, p1 );
    return det;
}

__forceinline__ __device__ RealType doOrient2DExact_Lifted
(
const RealType* p0,
const RealType* p1,
const RealType* p2,
bool            lifted,
RealType*       predConsts
)
{
    const RealType det = orient2dExact_Lifted( predConsts, p0, p1, p2, lifted);
    return det;
}

// Exact Incircle check must have failed (i.e. returned 0)
// No Infinity point here!!!
__forceinline__ __device__ Side doInCircleSoSOnly(
    const RealType* p0, const RealType* p1, const RealType* p2,
    const RealType* p3, int v0, int v1, int v2, int v3, RealType* predConsts )
{
    ////
    // Sort points using vertex as key, also note their sorted order
    ////

    const int NUM = DEG + 1;
    const RealType* p[NUM] = { p0, p1, p2, p3 };
    int pn = 1;

    if ( v0 > v2 ) { cuSwap( v0, v2 ); cuSwap( p[0], p[2] ); pn = -pn; }
    if ( v1 > v3 ) { cuSwap( v1, v3 ); cuSwap( p[1], p[3] ); pn = -pn; }
    if ( v0 > v1 ) { cuSwap( v0, v1 ); cuSwap( p[0], p[1] ); pn = -pn; }
    if ( v2 > v3 ) { cuSwap( v2, v3 ); cuSwap( p[2], p[3] ); pn = -pn; }
    if ( v1 > v2 ) { cuSwap( v1, v2 ); cuSwap( p[1], p[2] ); pn = -pn; }

    RealType result = 0;
    RealType pa2[2], pb2[2], pc2[2];
    int depth;

    for ( depth = 0; depth < 14; ++depth )
    {
        bool lifted = false;

        switch ( depth )
        {
        case 0:
            //printf("Here %i", depth);
            pa2[0] = p[1][0];   pa2[1] = p[1][1];
            pb2[0] = p[2][0];   pb2[1] = p[2][1];
            pc2[0] = p[3][0];   pc2[1] = p[3][1];
            break;
        case 1: lifted = true;
            //printf("Here %i", depth);
            //pa2[0] = p[1][0];   pa2[1] = p[1][1];
            //pb2[0] = p[2][0];   pb2[1] = p[2][1];
            //pc2[0] = p[3][0];   pc2[1] = p[3][1];
            break;
        case 2: lifted = true;
            //printf("Here %i", depth);
            pa2[0] = p[1][1];   pa2[1] = p[1][0];
            pb2[0] = p[2][1];   pb2[1] = p[2][0];
            pc2[0] = p[3][1];   pc2[1] = p[3][0];
            break;
        case 3:
            //printf("Here %i", depth);
            pa2[0] = p[0][0];   pa2[1] = p[0][1];
            pb2[0] = p[2][0];   pb2[1] = p[2][1];
            pc2[0] = p[3][0];   pc2[1] = p[3][1];
            break;
        case 4:
            //printf("Here %i", depth);
            result = p[2][0] - p[3][0];
            break;
        case 5:
            //printf("Here %i", depth);
            result = p[2][1] - p[3][1];
            break;
        case 6: lifted = true;
           // printf("Here %i\n", depth);
            //pa2[0] = p[0][0];   pa2[1] = p[0][1];
            //pb2[0] = p[2][0];   pb2[1] = p[2][1];
            //pc2[0] = p[3][0];   pc2[1] = p[3][1];
            break;
        case 7: lifted = true;
            //printf("Here %i\n", depth);
            pa2[0] = p[2][0];   pa2[1] = p[2][1];
            pb2[0] = p[3][0];   pb2[1] = p[3][1];
            break;
        case 8: lifted = true;
           // printf("Here %i\n", depth);
            pa2[0] = p[0][1];   pa2[1] = p[0][0];
            pb2[0] = p[2][1];   pb2[1] = p[2][0];
            pc2[0] = p[3][1];   pc2[1] = p[3][0];
            break;
        case 9:
            //printf("Here %i\n", depth);
            pa2[0] = p[0][0];   pa2[1] = p[0][1];
            pb2[0] = p[1][0];   pb2[1] = p[1][1];
            pc2[0] = p[3][0];   pc2[1] = p[3][1];
            break;
        case 10:
            //printf("Here %i\n", depth);
            result = p[1][0] - p[3][0];
            break;
        case 11:
           // printf("Here %i\n", depth);
            result = p[1][1] - p[3][1];
            break;
        case 12:
            //printf("Here %i\n", depth);
            result = p[0][0] - p[3][0];
            break;
        default:
           // printf("Here %i\n", depth);
            result = 1.0;
            break;
        }

        switch ( depth )
        {
        // 2D orientation determinant
        case 0: case 3: case 9:
        // 2D orientation involving the lifted coordinate
        case 1: case 2: case 6: case 8:
            result = doOrient2DExact_Lifted(
                pa2, pb2, pc2, lifted, predConsts );
            break;

        // 1D orientation involving the lifted coordinate
        case 7:
            result = doOrient1DExact_Lifted( pa2, pb2, predConsts );
            break;
        }

        if ( result != 0 )
            break;
    }

    switch ( depth )
    {
    case 1: case 3: case 5: case 8: case 10:
        result = -result;
    }

    const RealType det = result * pn;

    return cicToSide( det );
}


__forceinline__ __device__ Side doInCircleFastExactSoS(
Tri         tri,
int         vert,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts )
{
    if ( vert == infIdx )
        return SideOut;

    const RealType* pt[] = {
        getPoint( tri._v[0], points )._p,
        getPoint( tri._v[1], points )._p,
        getPoint( tri._v[2], points )._p,
        getPoint( vert, points )._p };

    if ( tri.has( infIdx ) )
    {
        const int infVi = tri.getIndexOf( infIdx );

        const Orient ort = doOrient2DFastExactSoS(
            tri._v[ (infVi + 1) % 3 ], tri._v[ (infVi + 2) % 3 ], vert,
            infIdx, points, orgPointIdx, predConsts );

        return cicToSide( ort );
    }

    const Side s0 = doInCircleFastExact(
        pt[0], pt[1], pt[2], pt[3], predConsts );

    if ( SideZero != s0 )
        return s0;

    // SoS
    if ( orgPointIdx != NULL )
    {
        tri._v[0] = orgPointIdx[ tri._v[0] ];
        tri._v[1] = orgPointIdx[ tri._v[1] ];
        tri._v[2] = orgPointIdx[ tri._v[2] ];
        vert      = orgPointIdx[ vert ];
    }

    const Side s1 = doInCircleSoSOnly(
        pt[0], pt[1], pt[2], pt[3], tri._v[0], tri._v[1], tri._v[2], vert,
        predConsts );

    return s1;
}


template < CheckDelaunayMode checkMode >
__forceinline__ __device__ void
checkDelaunayFast
(
int*    actTriArr,
Tri*    triArr,
TriOpp* oppArr,
char*   triInfoArr,
int*    triVoteArr,
int2*   exactCheckVi,
int     actTriNum,
int*    counter,
int*    dbgCircleCountArr,
/** predWrapper values **/
int         infIdx,
Point2*     points,
RealType*   predConsts
)
{
    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < actTriNum; idx += getThreadNum() )
    {
        const int botTi = actTriArr[ idx ];

        ////
        // Check which side needs to be checked
        ////
        int checkVi         = 1;
        const TriOpp botOpp = oppArr[ botTi ];

        for ( int botVi = 0; botVi < DEG; ++botVi )
            if ( -1 != botOpp._t[ botVi ]         // No neighbour at this face
                && !botOpp.isOppConstraint( botVi ) )   // or is a constraint
            {
                const int topTi = botOpp.getOppTri( botVi );
                const int topVi = botOpp.getOppVi( botVi );

                if ( ( ( botTi < topTi ) ||
                        Checked == getTriCheckState( triInfoArr[ topTi ] ) ) )
                    checkVi = (checkVi << 2) | botVi;
            }

        // Nothing to check?
        if ( checkVi != 1 )
        {
            ////
            // Do circle check
            ////
            const Tri botTri = triArr[ botTi ];

            int dbgCount = 0;
            bool hasFlip = false;
            int exactVi  = 1;

            // Check 2-3 flips
            for ( ; checkVi > 1; checkVi >>= 2 )
            {
                const int botVi   = ( checkVi & 3 );
                const int topTi   = botOpp.getOppTri( botVi );
                const int topVi   = botOpp.getOppVi( botVi );

                const int topVert = triArr[ topTi ]._v[ topVi ];

                Side side = doInCircleFast(
                    botTri, topVert, infIdx, points, predConsts );

                ++dbgCount;

                if ( SideZero == side )
                    if ( checkMode == CircleFastOrientFast )
                        // Store for future exact mode
                        oppArr[ botTi ].setOppSpecial( botVi, true );
                    else
                        // Pass to next kernel - exact kernel
                        exactVi = (exactVi << 2) | botVi;

                // No incircle failure at this face
                if ( SideIn != side ) continue;

                // We have incircle failure, vote!
                voteForFlip( triVoteArr, botTi, topTi, botVi );
                hasFlip = true;
                break;
            }

            if ( ( checkMode == CircleExactOrientSoS ) &&
                    ( !hasFlip ) && ( exactVi != 1 ) )
                storeIntoBuffer( exactCheckVi, &counter[ CounterExact ],
                                 make_int2( botTi, exactVi ) );

            if ( NULL != dbgCircleCountArr )
                dbgCircleCountArr[ botTi ] = dbgCount;
        }
    }

    return;
}

__global__ void
kerCheckDelaunayFast
(
int*    actTriArr,
Tri*    triArr,
TriOpp* oppArr,
char*   triInfoArr,
int*    triVoteArr,
int     actTriNum,
/** predWrapper values **/
int         infIdx,
Point2*     points,
RealType*   predConsts
)
{
    checkDelaunayFast< CircleFastOrientFast >(
        actTriArr,
        triArr,
        oppArr,
        triInfoArr,
        triVoteArr,
        NULL,
        actTriNum,
        NULL,
        NULL,
        infIdx,
        points,
        predConsts );
    return;
}

__global__ void
kerCheckDelaunayExact_Fast
(
int*    actTriArr,
Tri*    triArr,
TriOpp* oppArr,
char*   triInfoArr,
int*    triVoteArr,
int2*   exactCheckVi,
int     actTriNum,
int*    counter,
/** predWrapper values **/
int         infIdx,
Point2*     points,
RealType*   predConsts
)
{
    checkDelaunayFast< CircleExactOrientSoS >(
        actTriArr,
        triArr,
        oppArr,
        triInfoArr,
        triVoteArr,
        exactCheckVi,
        actTriNum,
        counter,
        NULL,
        infIdx,
        points,
        predConsts );
    return;
}

__global__ void
kerCheckDelaunayExact_Exact
(
Tri*    triArr,
TriOpp* oppArr,
int*    triVoteArr,
int2*   exactCheckVi,
int*    counter,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts
)
{
    int* dbgCircleCountArr = NULL;
    const int exactNum = counter[ CounterExact ];

    // Iterate active triangle
    for ( int idx = getCurThreadIdx(); idx < exactNum; idx += getThreadNum() )
    {
        int2 val    = exactCheckVi[ idx ];
        int botTi   = val.x;
        int exactVi = val.y;

        exactCheckVi[ idx ] = make_int2( -1, -1 );

        ////
        // Do circle check
        ////
        TriOpp botOpp    = oppArr[ botTi ];
        const Tri botTri = triArr[ botTi ];

        int dbgCount    = 0;

        if ( NULL != dbgCircleCountArr )
            dbgCount = dbgCircleCountArr[ botTi ];

        for ( ; exactVi > 1; exactVi >>= 2 )
        {
            const int botVi = ( exactVi & 3 );

            const int topTi     = botOpp.getOppTri( botVi );
            const int topVi     = botOpp.getOppVi( botVi );
            const int topVert   = triArr[ topTi ]._v[ topVi ];

            const Side side = doInCircleFastExactSoS(
                botTri, topVert, infIdx, points, orgPointIdx, predConsts );

            ++dbgCount;

            if ( SideIn != side ) continue; // No incircle failure at this face

            voteForFlip( triVoteArr, botTi, topTi, botVi );
            break;
        } // Check faces of triangle

        if ( NULL != dbgCircleCountArr )
            dbgCircleCountArr[ botTi ] = dbgCount;
    }

    return;
}

// Note: triVoteArr should *not* be modified here
__global__ void
kerMarkRejectedFlips
(
int*        actTriArr,
TriOpp*     oppArr,
int*        triVoteArr,
char*       triInfoArr,
int*        flipToTri,
int         actTriNum
)
{
    int* dbgRejFlipArr = NULL;
    for ( int idx = getCurThreadIdx(); idx < actTriNum; idx += getThreadNum() )
    {
        int output = -1;

        const int triIdx  = actTriArr[ idx ];
        const int voteVal = triVoteArr[ triIdx ];

        if ( INT_MAX == voteVal )
        {
            setTriCheckState( triInfoArr[ triIdx ], Checked );
            actTriArr[ idx ] = -1;
        }
        else
        {
            int bossTriIdx, botVi;

            decode( voteVal, &bossTriIdx, &botVi );

            if ( bossTriIdx == triIdx ) // Boss of myself
            {
                const TriOpp& opp    = oppArr[ triIdx ];
                const int topTriIdx  = opp.getOppTri( botVi );
                const int topVoteVal = triVoteArr[ topTriIdx ];

                if ( topVoteVal == voteVal )
                    output = voteVal;
            }

            if ( NULL != dbgRejFlipArr && output == -1 )
                dbgRejFlipArr[ triIdx ] = 1;
        }

        flipToTri[ idx ] = output;
    }

    return;
}

struct FlipItem {
    int _v[2];
    int _t[2];
};

struct FlipItemTriIdx {
    int _t[2];
};

__forceinline__ __device__
void storeFlip( FlipItem* flipArr, int idx, const FlipItem& item )
{
    int4 t = { item._v[0], item._v[1], item._t[0], item._t[1] };

    ( ( int4 * ) flipArr )[ idx ] = t;
}

__forceinline__ __device__
FlipItem loadFlip( FlipItem* flipArr, int idx )
{
    int4 t = ( ( int4 * ) flipArr )[ idx ];

    FlipItem flip = { t.x, t.y, t.z, t.w };

    return flip;
}

// idx    Constraint index
// vi     The vertex opposite the next intersected edge. vi = 3 if this is the
//         last triangle
//        --> vi+1 is on the right, vi+2 is on the left of the constraint
// side   Which side of the constraint the vertex vi lies on; 0-cw, 1-ccw,
//        2-start, 3-end
__forceinline__ __device__ int encode_constraint( int idx, int vi, int side )
{
    return ( idx << 4 ) | ( vi << 2 ) | side;
}

__forceinline__ __device__ int decode_cIdx( int label )
{
    return ( label >> 4 );
}

__forceinline__ __device__ int decode_cVi( int label )
{
    return ( label >> 2 ) & 3;
}

__forceinline__ __device__ int decode_cSide( int label )
{
    return ( label & 3);
}

__global__ void
kerFlip
(
int*        flipToTri,
int         nFlip,
Tri*        triArr,
TriOpp*     oppArr,
char*       triInfoArr,
int2*       triMsgArr,
int*        actTriArr,
FlipItem*   flipArr,
int         orgFlipNum,
int         actTriNum,
int         useActTri
)
{
    int*        triConsArr = NULL;
    int*        vertTriArr = NULL;

    if(!useActTri) {
        actTriArr = NULL;
    }

    // Iterate flips
    for ( int flipIdx = getCurThreadIdx(); flipIdx < nFlip;
          flipIdx += getThreadNum() )
    {
        int botIdx, botVi;

        const int voteVal = flipToTri[ flipIdx ];

        decode( voteVal, &botIdx, &botVi );

        // Bottom triangle
        Tri botTri            = triArr[ botIdx ];
        const TriOpp& botOpp  = oppArr[ botIdx ];

        // Top triangle
        const int topIdx = botOpp.getOppTri( botVi );
        const int topVi  = botOpp.getOppVi( botVi );
        Tri topTri       = triArr[ topIdx ];

        const int globFlipIdx = orgFlipNum + flipIdx;

        const int botAVi = ( botVi + 1 ) % 3;
        const int botBVi = ( botVi + 2 ) % 3;
        const int topAVi = ( topVi + 2 ) % 3;
        const int topBVi = ( topVi + 1 ) % 3;

        // Create new triangle
        const int topVert = topTri._v[ topVi ];
        const int botVert = botTri._v[ botVi ];
        const int botA    = botTri._v[ botAVi ];
        const int botB    = botTri._v[ botBVi ];

        // Update the bottom and top triangle
        botTri = makeTri( botVert, botA, topVert );
        topTri = makeTri( topVert, botB, botVert );

        triArr[ botIdx ] = botTri;
        triArr[ topIdx ] = topTri;

        int newBotNei = 0xffff;
        int newTopNei = 0xffff;

        setTriIdxVi( newBotNei, botAVi, 1, 0 );
        setTriIdxVi( newBotNei, botBVi, 3, 2 );
        setTriIdxVi( newTopNei, topAVi, 3, 2 );
        setTriIdxVi( newTopNei, topBVi, 0, 0 );

        // Write down the new triangle idx
        triMsgArr[ botIdx ] = make_int2( newBotNei, globFlipIdx );
        triMsgArr[ topIdx ] = make_int2( newTopNei, globFlipIdx );

        // Record the flip
        FlipItem flipItem = { botVert, topVert, botIdx, topIdx };
        storeFlip( flipArr, globFlipIdx, flipItem );

        // Prepare for the next round
        if ( actTriArr != NULL )
            actTriArr[ actTriNum + flipIdx ] =
                ( Checked == getTriCheckState( triInfoArr[ topIdx ] ) )
                ? topIdx : -1;

        if ( triConsArr == NULL )       // Standard flipping
            triInfoArr[ topIdx ] = 3;  // Alive + Changed
        else
        {
            vertTriArr[ botA ] = botIdx;
            vertTriArr[ botB ] = topIdx;

            // Update constraint intersection info
            int botLabel = triConsArr[ botIdx ];
            int topLabel = triConsArr[ topIdx ];

            const int consIdx   = decode_cIdx( botLabel );
            const int botSide   = decode_cSide( botLabel );
                  int topSide   = decode_cSide( topLabel );

            if ( topSide < 2 )  // Not the last triangle
                topSide = ( decode_cVi( topLabel ) == topAVi ? 0 : 1 );

            switch ( botSide )      // Cannto be 3
            {
            case 0:
                switch ( topSide )
                {
                case 0:
                    botLabel = -1;
                    topLabel = encode_constraint( consIdx, 2, 0 );
                    break;
                case 1:
                    botLabel = encode_constraint( consIdx, 0, 0 );
                    topLabel = encode_constraint( consIdx, 1, 1 );
                    break;
                case 3:
                    botLabel = -1;
                    topLabel = encode_constraint( consIdx, 0, 3 );
                    break;
                }
                break;
            case 1:
                switch ( topSide )
                {
                case 0:
                    botLabel = encode_constraint( consIdx, 1, 0 );
                    topLabel = encode_constraint( consIdx, 2, 1 );
                    break;
                case 1:
                    botLabel = encode_constraint( consIdx, 0, 1 );
                    topLabel = -1;
                    break;
                case 3:
                    botLabel = encode_constraint( consIdx, 2, 3 );
                    topLabel = -1;
                    break;
                }
                break;
            case 2:
                botLabel = ( topSide == 1 ? encode_constraint(
                    consIdx, 0, 2 ) : -1 );
                topLabel = ( topSide == 0 ? encode_constraint(
                    consIdx, 2, 2 ) : -1 );
                break;
            }

            triConsArr[ botIdx ] = botLabel;
            triConsArr[ topIdx ] = topLabel;
        }
    }

    return;
}

__global__ void
kerUpdateOpp
(
FlipItem*    flipVec,
TriOpp*      oppArr,
int2*        triMsgArr,
int*         flipToTri,
int          orgFlipNum,
int          flipNum
)
{
    // Iterate flips
    for ( int flipIdx = getCurThreadIdx(); flipIdx < flipNum;
            flipIdx += getThreadNum() )
    {
        int botIdx, botVi;

        int voteVal = flipToTri[ flipIdx ];

        decode( voteVal, &botIdx, &botVi );

        int     extOpp[4];
        TriOpp  opp;

        opp = oppArr[ botIdx ];

        extOpp[ 0 ] = opp.getOppTriVi( (botVi + 1) % 3 );
        extOpp[ 1 ] = opp.getOppTriVi( (botVi + 2) % 3 );

        int topIdx      = opp.getOppTri( botVi );
        const int topVi = opp.getOppVi( botVi );

        opp = oppArr[ topIdx ];

        extOpp[ 2 ] = opp.getOppTriVi( (topVi + 2) % 3 );
        extOpp[ 3 ] = opp.getOppTriVi( (topVi + 1) % 3 );

        // Ok, update with neighbors
        for ( int i = 0; i < 4; ++i )
        {
            int newTriIdx, vi;
            int triOpp  = extOpp[ i ];
            bool isCons = isOppValConstraint( triOpp );

            // No neighbor
            if ( -1 == triOpp ) continue;

            int oppIdx = getOppValTri( triOpp );
            int oppVi  = getOppValVi( triOpp );

            const int2 msg = triMsgArr[ oppIdx ];

            if ( msg.y < orgFlipNum )    // Neighbor not flipped
            {
                // Set my neighbor's opp
                newTriIdx = ( (i & 1) == 0 ? topIdx : botIdx );
                vi        = ( i == 0 || i == 3 ) ? 0 : 2;

                oppArr[ oppIdx ].setOpp( oppVi, newTriIdx, vi, isCons );
            }
            else
            {
                const int oppFlipIdx = msg.y - orgFlipNum;

                // Update my own opp
                const int newLocOppIdx = getTriIdx( msg.x, oppVi );

                if ( newLocOppIdx != 3 )
                    oppIdx = flipVec[ oppFlipIdx ]._t[ newLocOppIdx ];

                oppVi = getTriVi( msg.x, oppVi );

                setOppValTriVi( extOpp[ i ], oppIdx, oppVi );
            }
        }

        // Now output
        opp._t[ 0 ] = extOpp[ 3 ];
        opp.setOpp( 1, topIdx, 1 );
        opp._t[ 2 ] = extOpp[ 1 ];

        oppArr[ botIdx ] = opp;

        opp._t[ 0 ] = extOpp[ 0 ];
        opp.setOpp( 1, botIdx, 1 );
        opp._t[ 2 ] = extOpp[ 2 ];

        oppArr[ topIdx ] = opp;
    }

    return;
}

__global__ void
kerUpdateFlipTrace
(
FlipItem*   flipArr,
int*        triToFlip,
int         orgFlipNum,
int         flipNum
)
{
    for ( int idx = getCurThreadIdx(); idx < flipNum; idx += getThreadNum() )
    {
        const int flipIdx   = orgFlipNum + idx;
        FlipItem flipItem   = loadFlip( flipArr, flipIdx );

        int triIdx, nextFlip;

        triIdx              = flipItem._t[ 0 ];
        nextFlip            = triToFlip[ triIdx ];
        flipItem._t[ 0 ]    = (
            ( nextFlip == -1 ) ? ( triIdx << 1 ) | 0 : nextFlip);
        triToFlip[ triIdx ] = ( flipIdx << 1 ) | 1;

        triIdx              = flipItem._t[ 1 ];
        nextFlip            = triToFlip[ triIdx ];
        flipItem._t[ 1 ]    = (
            ( nextFlip == -1 ) ? ( triIdx << 1 ) | 0 : nextFlip);
        triToFlip[ triIdx ] = ( flipIdx << 1 ) | 1;

        storeFlip( flipArr, flipIdx, flipItem );
    }
}

template<bool doFast>
__forceinline__ __device__ bool
relocatePoints
(
int         vertex,
int&        location,
int*        triToFlip,
FlipItem*   flipArr,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts
)
{
    const int triIdx = location;

    if ( triIdx == -1 ) return true;

    int nextIdx = ( doFast ) ? triToFlip[ triIdx ] : triIdx;

    if ( nextIdx == -1 ) return true;   // No flip

    int flag              = nextIdx & 1;
    int destIdx           = nextIdx >> 1;

    while ( flag == 1 )
    {
        const FlipItem flipItem = loadFlip( flipArr, destIdx );

        const Orient ord = doFast
            ? doOrient2DFast(
                flipItem._v[ 0 ], flipItem._v[ 1 ], vertex,
                infIdx, points, predConsts)
            : doOrient2DFastExactSoS(
                flipItem._v[ 0 ], flipItem._v[ 1 ], vertex, infIdx,
                points, orgPointIdx, predConsts );

        if ( doFast && ( OrientZero == ord ) )
        {
            location = nextIdx;
            return false;
        }

        nextIdx = flipItem._t[ ( OrientPos == ord ) ? 1 : 0 ];
        flag    = nextIdx & 1;
        destIdx = nextIdx >> 1;
    }

    location = destIdx; // Write back

    return true;
}

__global__ void
kerRelocatePointsFast
(
int*        vertTriVec,
int         nVertTri,
int*        triToFlip,
FlipItem*   flipArr,
int*        exactCheckArr,
int*        counter,
/** predWrapper values **/
int         infIdx,
Point2*     points,
int*        orgPointIdx,
RealType*   predConsts
)
{
    // Iterate points
    for ( int idx = getCurThreadIdx(); idx < nVertTri; idx += getThreadNum() )
    {
        bool ret = relocatePoints<true>(
            idx, vertTriVec[ idx ], triToFlip, flipArr, infIdx, points,
            orgPointIdx, predConsts );

        if ( !ret )
            storeIntoBuffer( exactCheckArr, &counter[ CounterExact ], idx );
    }
}

__global__ void
kerRelocatePointsExact
(
int*        vertTriArr,
int*        triToFlip,
FlipItem*   flipArr,
int*        exactCheckArr,
int*        counter,
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

        relocatePoints< false >(
            vertIdx, vertTriArr[ vertIdx ], triToFlip, flipArr, infIdx, points,
            orgPointIdx, predConsts );
    }
}

__global__ void
kerMarkInfinityTri
(
Tri*        triVec,
int         nTri,
char*       triInfoArr,
TriOpp*     oppArr,
int         infIdx
)
{
    for ( int idx = getCurThreadIdx(); idx < nTri; idx += getThreadNum() )
    {
        if ( !triVec[ idx ].has( infIdx ) ) continue;

        // Mark as deleted
        setTriAliveState( triInfoArr[ idx ], false );

        TriOpp opp = oppArr[ idx ];

        for ( int vi = 0; vi < DEG; ++vi )
        {
            if ( opp._t[ vi ] < 0 ) continue;

            const int oppIdx = opp.getOppTri( vi );
            const int oppVi  = opp.getOppVi( vi );

            oppArr[ oppIdx ]._t[ oppVi ] = -1;
        }
    }
}

__global__ void
kerCollectFreeSlots
(
char* triInfoArr,
int*  prefixArr,
int*  freeArr,
int   newTriNum
)
{
    for ( int idx = getCurThreadIdx(); idx < newTriNum; idx += getThreadNum() )
    {
        if ( isTriAlive( triInfoArr[ idx ] ) ) continue;

        int freeIdx = idx - prefixArr[ idx ];

        freeArr[ freeIdx ] = idx;
    }
}

__global__ void
kerMakeCompactMap
(
char*        triInfoVec,
int          nTriInfo,
int*         prefixArr,
int*         freeArr,
int          newTriNum
)
{
    for ( int idx = newTriNum + getCurThreadIdx();
          idx < nTriInfo; idx += getThreadNum() )
    {
        if ( !isTriAlive( triInfoVec[ idx ] ) )
        {
            prefixArr[ idx ] = -1;
            continue;
        }

        int freeIdx     = newTriNum - prefixArr[ idx ];
        int newTriIdx   = freeArr[ freeIdx ];

        prefixArr[ idx ] = newTriIdx;
    }
}

__global__ void
kerCompactTris
(
char*        triInfoVec,
int          nTriInfo,
int*         prefixArr,
Tri*         triArr,
TriOpp*      oppArr,
int          newTriNum
)
{
    for ( int idx = newTriNum + getCurThreadIdx();
          idx < nTriInfo; idx += getThreadNum() )
    {
        int newTriIdx = prefixArr[ idx ];

        if ( newTriIdx == -1 ) continue;

        triArr[ newTriIdx ]          = triArr[ idx ];
        triInfoVec[ newTriIdx ] = triInfoVec[ idx ];

        TriOpp opp = oppArr[ idx ];

        for ( int vi = 0; vi < DEG; ++vi )
        {
            if ( opp._t[ vi ] < 0 ) continue;

            const int oppIdx = opp.getOppTri( vi );

            if ( oppIdx >= newTriNum )
            {
                const int oppNewIdx = prefixArr[ oppIdx ];

                opp.setOppTri( vi, oppNewIdx );
            }
            else
            {
                const int oppVi = opp.getOppVi( vi );

                oppArr[ oppIdx ].setOppTri( oppVi, newTriIdx );
            }
        }

        oppArr[ newTriIdx ] = opp;
    }
}

__global__ void
kerUpdateVertIdx
(
Tri*        triVec,
int         nTri,
char*       triInfoArr,
int*        orgPointIdx
)
{
    for ( int idx = getCurThreadIdx(); idx < nTri; idx += getThreadNum() )
    {
        if ( !isTriAlive( triInfoArr[ idx ] ) ) continue;

        Tri tri = triVec[ idx ];

        for ( int i = 0; i < DEG; ++i )
            tri._v[ i ] = orgPointIdx[ tri._v[i] ];

        triVec[ idx ] = tri;
    }
}

__device__ bool isPointInTriangle(
        const Point2 p, const Point2 p0, const Point2 p1, const Point2 p2,
        RealType* s, RealType* t, double eps) {

    RealType A = 0.5 * (
        (-p1._p[1]) * p2._p[0] +
        p0._p[1] * (-p1._p[0] + p2._p[0]) +
        p0._p[0] * (p1._p[1] - p2._p[1]) +
        p1._p[0] * p2._p[1]);

    if(A <= 1e-13) {
        // Omit close to colinear triangles
        return false;
    }

    RealType sign = A < 0 ? -1 : 1;
    RealType unS = (p0._p[1] * p2._p[0] - p0._p[0] * p2._p[1] +
          (p2._p[1] - p0._p[1]) * p._p[0] +
          (p0._p[0] - p2._p[0]) * p._p[1]) * sign;

    RealType unT = (p0._p[0] * p1._p[1] - p0._p[1] * p1._p[0] +
          (p0._p[1] - p1._p[1]) * p._p[0] +
          (p1._p[0] - p0._p[0]) * p._p[1]) * sign;

    *s = 1.0 / (2.0 * A) * unS;
    *t = 1.0 / (2.0 * A) * unT;

    if(eps != 0.0) {
        return (unS >= 0 &&
                unT >= 0 &&
                ((unS + unT + eps) <= 2 * A * sign ||
                 (unS + unT - eps) <= 2 * A * sign));
    }
    return unT >= 0 && unS >= 0 && unS + unT <= 2 * A * sign;
}

__global__ void getMortonNumber(
        const double* points, const int n_points, const double* min_val,
        const double* range, int* out) {

    for ( int idx = getCurThreadIdx(); idx < n_points; idx += getThreadNum() )
    {
        const double* point = points + 2 * idx;

        // Creates 16-bit gap between value bits
        const int Gap08 = 0x00FF00FF;
        const int Gap04 = 0x0F0F0F0F;   // ... and so on ...
        const int Gap02 = 0x33333333;   // ...
        const int Gap01 = 0x55555555;   // ...

        const int minInt = 0x0;
        const int maxInt = 0x7FFF;

        int mortonNum = 0;

        // Iterate coordinates of point
        for ( int vi = 0; vi < 2; ++vi )
        {
            // Read
            int v = int( ( point[ vi ] - min_val[0] ) / range[0] * 32768.0 );

            if ( v < minInt )
                v = minInt;

            if ( v > maxInt )
                v = maxInt;

            // Create 1-bit gaps between the 10 value bits
            // Ex: 1010101010101010101
            v = ( v | ( v <<  8 ) ) & Gap08;
            v = ( v | ( v <<  4 ) ) & Gap04;
            v = ( v | ( v <<  2 ) ) & Gap02;
            v = ( v | ( v <<  1 ) ) & Gap01;

            // Interleave bits of x-y coordinates
            mortonNum |= ( v << vi );
        }

        out[idx] = mortonNum;
    }
}

__global__ void computeDistance2D(
        const double* points, const int n_points, const long long* a_idx,
        const long long* b_idx, int* out) {

    for ( int idx = getCurThreadIdx(); idx < n_points; idx += getThreadNum() )
    {
        const double* p_c = points + 2 * idx;
        const double* p_a = points + 2 * a_idx[0];
        const double* p_b = points + 2 * b_idx[0];

        double abx = p_b[0] - p_a[0];
        double aby = p_b[1] - p_a[1];

        double acx = p_c[0] - p_a[0];
        double acy = p_c[1] - p_a[1];

        double dist = abx * acy - aby * acx;
        int int_dist = __float_as_int( fabs((float) dist) );
        out[idx] = int_dist;
    }
}

__global__ void makeKeyFromTriHasVert
(
int* triHasVert,
int  nTri,
int* out
)
{
    for ( int idx = getCurThreadIdx(); idx < nTri; idx += getThreadNum() ) {
        out[idx] = triHasVert[idx] < INT_MAX - 1 ? 2 : 0;
    }
}

#define DBL_EPSILON     2.220440492503130e-16

__global__ void kerCheckIfCoplanarPoints
(
RealType* points,
long long*       paIdx,
long long*       pbIdx,
long long*       pcIdx,
RealType* det
) {
    RealType* pa = points + 2 * *paIdx;
    RealType* pb = points + 2 * *pbIdx;
    RealType* pc = points + 2 * *pcIdx;

    RealType detLeft = (pa[0] - pc[0]) * (pb[1] - pc[1]);
    RealType detRight = (pa[1] - pc[1]) * (pb[0] - pc[0]);
    RealType detFull = detLeft - detRight;

    bool isDetLeftPos = detLeft > 0;
    bool isDetLeftNeg = detLeft < 0;

    bool isDetRightPos = detRight >= 0;
    bool isDetRightNeg = detRight <= 0;

    bool inspectMore = false;
    RealType detSum = 0;

    if(isDetLeftPos) {
        if(isDetRightNeg) {
            *det = detFull;
        } else {
            detSum = detLeft + detRight;
            inspectMore = true;
        }
    } else if (isDetLeftNeg) {
        if(isDetRightPos) {
            *det = detFull;
        } else {
            detSum = -detLeft - detRight;
            inspectMore = true;
        }
    } else {
        *det = detFull;
    }

    if(!inspectMore) {
        return;
    }

    RealType ccwerrboundA = (3.0 + 16.0 * DBL_EPSILON) * DBL_EPSILON;
    RealType errBound = ccwerrboundA * detSum;

    if(detFull >= errBound || -detFull >= errBound ) {
        *det = detFull;
    }

    *det = 0;
}

__device__ unsigned int zCurvePoint2
(
Point2            center,
const double*     minVal,
const double*     range
) {
    unsigned int encoding = 0;

    const int minInt = 0x0;
    const int maxInt = 0x7fff;

    for(int i = 0; i < 2; i++) {
        RealType v = center._p[i];
        int xV = ( ( v - minVal[0] ) / range[0] * 32768.0 );

        if ( xV < 0 )
            xV = 0;

        if ( xV > maxInt )
            xV = maxInt;

        unsigned int x = static_cast<unsigned int>(xV);

        x &= 0x7fff;
        x = (x | (x << 8)) & 0x7f00ff;
        x = (x | (x << 4)) & 0x70f0f0f;
        x = (x | (x << 2)) & 0x13333333;
        x = (x | (x << 1)) & 0x15555555;

        encoding |= (x << (1 - i));
    }

    return encoding;
}

__device__ int zCurveBisect
(
unsigned int        query,
long long*          encIdx,
unsigned int*       triEnc,
int                 nTri,
bool                leftEnd
) {
    int left = 0;
    int right = nTri - 1;

    bool indirect = encIdx != NULL;

    while(left < right) {
        int mid = (left + right) / 2;
        long long midIdx = indirect ? encIdx[mid] : mid;
        unsigned int midTri = triEnc[midIdx];

        if(midTri > query) {
            right = mid - 1;
        } else if(query > midTri) {
            left = mid + 1;
        } else {
            left = mid;
            right = mid;
        }
    }

    return leftEnd ? left : right + 1;
}

__global__ void kerEncodeEdges
(
Tri*          tri,
int           nTri,
Point2*       triPoints,
RealType*     minVal,
RealType*     range,
unsigned int* edgeEnc,
int*          edges
)
{
    for ( int idx = getCurThreadIdx(); idx < nTri; idx += getThreadNum() ) {
        Tri curTri = tri[idx];
        // int* curEdge = edges + 2 * idx;

        int prevVi = curTri._v[0];
        Point2 prevP = triPoints[prevVi];
        for( int i = 1; i <= DEG; i++ ) {
            int vi = curTri._v[i % DEG];

            Point2 curP = triPoints[vi];
            Point2 mid = Point2{{
                (prevP._p[0] + curP._p[0]) / 2.0,
                (prevP._p[1] + curP._p[1]) / 2.0
            }};

            unsigned int enc = zCurvePoint2(mid, minVal, range);
            edgeEnc[3 * idx + (i - 1)] = enc;
            edges[6 * idx + 2 * (i - 1)] = prevVi;
            edges[6 * idx + 2 * (i - 1) + 1] = vi;

            prevP = curP;
            prevVi = vi;

        }
    }
}

__global__ void kerCountVertexNeighbors
(
int*          edges,
int           nEdges,
int*          vertexCount,
int           nVertices
) {
    for ( int idx = getCurThreadIdx(); idx < nEdges; idx += getThreadNum() ) {
        int* edge = edges + 2 * idx;
        int fromVi = edge[0];
        int toVi = edge[1];

        int pos1 = atomicAdd(vertexCount + fromVi, 1);
        int pos2 = atomicAdd(vertexCount + toVi, 1);
    }
}

__global__ void kerFillVertexNeighbors
(
int*             edges,
int              nEdges,
long long*       vertexOff,
int*             vertexCount,
int*             vertexNeighbors
) {
    for ( int idx = getCurThreadIdx(); idx < nEdges; idx += getThreadNum() ) {
        int* edge = edges + 2 * idx;
        int fromVi = edge[0];
        int toVi = edge[1];

        int pos = atomicSub(vertexCount + fromVi, 1);
        vertexNeighbors[vertexOff[fromVi] + pos - 1] = toVi;

        int pos2 = atomicSub(vertexCount + toVi, 1);
        vertexNeighbors[vertexOff[toVi] + pos2 - 1] = fromVi;

    }
}

__global__ void kerEncBarycenters
(
Tri*                tri,
int                 nTri,
Point2*             points,
const double*       minVal,
const double*       range,
unsigned int*       out,
Point2*             centers
) {
    for ( int idx = getCurThreadIdx(); idx < nTri; idx += getThreadNum()) {
        Tri curTri = tri[idx];
        RealType mx = 0;
        RealType my = 0;

        // Point2 prevPoint = points[curTri._v[0]];
        for( int i = 0; i < DEG; i++ ) {
            Point2 curPoint = points[curTri._v[i]];
            mx += curPoint._p[0];
            my += curPoint._p[1];
        }

        mx = mx / 3.0;
        my = my / 3.0;

        Point2 center = Point2{ { mx, my } };
        out[idx] = zCurvePoint2(center, minVal, range);
        centers[idx] = center;
    }
}

__device__ bool checkPointInTriangle
(
int        i,
Tri*       tri,
long long* encIdx,
Point2*    triPoints,
Point2     query,
RealType*  s,
RealType*  t,
double     eps,
bool       indirect
) {
    int search_idx = indirect ? encIdx[i] : i;
    Tri curTri = tri[search_idx];

    Point2 p0 = triPoints[curTri._v[0]];
    Point2 p1 = triPoints[curTri._v[1]];
    Point2 p2 = triPoints[curTri._v[2]];

    return isPointInTriangle(query, p0, p1, p2, s, t, eps);
}


__device__ bool isQueryInRange
(
Point2 query,
Point2 minRange,
Point2 maxRange
) {
    return (query._p[0] >= minRange._p[0] &&
            query._p[0] <= maxRange._p[0] &&
            query._p[1] >= minRange._p[1] &&
            query._p[1] <= maxRange._p[1]);
}

__device__ double computePointDist
(
Point2 a,
Point2 b
) {
    double xDiff = a._p[0] - b._p[0];
    double yDiff = a._p[1] - b._p[1];

    return xDiff * xDiff + yDiff * yDiff;
}

__device__ double dot(Point2 a, Point2 b) {
    return a._p[0] * b._p[0] + a._p[1] * b._p[1];
}

__device__ Point2 findClosestPointToTri
(
Point2 p,
Point2 a,
Point2 b,
Point2 c
) {
    const Point2 ab = b - a;
    const Point2 ac = c - a;
    const Point2 ap = p - a;

    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.f && d2 <= 0.f) return a; //#1

    const Point2 bp = p - b;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.f && d4 <= d3) return b; //#2

    const Point2 cp = p - c;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.f && d5 <= d6) return c; //#3

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const double v = d1 / (d1 - d3);
        return a + ab * v; //#4
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const double v = d2 / (d2 - d6);
        return a + ac * v; //#5
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        const double v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * v; //#6
    }

    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom;
    const double w = vc * denom;
    return a + ab * v + ac * w; //#0
}

__device__ double computePointTriDist
(
int     i,
Point2  query,
Point2* centers,
Tri*    tri,
Point2* triPoints,
double* centerDist
) {
    Tri curTri = tri[i];
    Point2 center = centers[i];

    Point2 p0 = triPoints[curTri._v[0]];
    Point2 p1 = triPoints[curTri._v[1]];
    Point2 p2 = triPoints[curTri._v[2]];

    Point2 closestToTri = findClosestPointToTri(query, p0, p1, p2);
    *centerDist = computePointDist(query, center);
    return computePointDist(query, closestToTri);

}

__device__ bool visitedAlready(int v, int* visited, int maxLen) {
    bool visitFound = false;
    for(int i = 0; i < maxLen && !visitFound; i++) {
        visitFound = visited[i] == v;
    }
    return visitFound;
}

__global__ void kerFindClosestTri
(
Point2*             queries,
int                 nQueries,
Tri*                tri,
int                 nTri,
TriOpp*             triOpp,
long long*          encIdx,
unsigned int*       triEnc,
Point2*             triPoints,
Point2*             centers,
const double*       minVal,
const double*       range,
Point2*             minAxis,
Point2*             maxAxis,
double              eps,
bool                findCoords,
int*                out,
RealType*           coords
) {
    // For debugging
    // int* debug = NULL;

    for (int idx = getCurThreadIdx(); idx < nQueries; idx += getThreadNum()) {
        Point2 query = queries[idx];

        if(!isQueryInRange(query, minAxis[0], maxAxis[0])) {
            out[idx] = -1;
            continue;
        }

        unsigned int enc = zCurvePoint2(query, minVal, range);

        int off = 0;
        int pos = zCurveBisect(enc, encIdx, triEnc, nTri, true);

        bool isInTri = false;
        int trianglePos = -1;
        int stoppedAt = -1;
        RealType s = -1.0;
        RealType t = -1.0;

        int startingPos = pos;
        double minDist = CUDART_INF;
        double minCenterDist = CUDART_INF;

        // Check if query is inside the triangle represented by the found
        // center or a radius of two
        isInTri = checkPointInTriangle(pos, tri, encIdx, triPoints,
                                       query, &s, &t, eps, true);

        if(isInTri) {
            trianglePos = encIdx[pos];
            stoppedAt = 0;
        } else {
            minDist = computePointTriDist(
                encIdx[pos], query, centers, tri, triPoints, &minCenterDist);
        }

        if(!isInTri && pos + 1 < nTri) {
            isInTri = checkPointInTriangle(pos + 1, tri, encIdx, triPoints,
                                            query, &s, &t, eps, true);

            if(isInTri) {
                trianglePos = encIdx[pos + 1];
                stoppedAt = 0;
            } else {
                double centerDist;
                double dist = computePointTriDist(
                    encIdx[pos + 1], query, centers, tri, triPoints,
                    &centerDist);
                if(dist < minDist) {
                    minDist = dist;
                    startingPos = pos + 1;
                } else if(dist == minDist && centerDist < minCenterDist) {
                    minDist = dist;
                    minCenterDist = centerDist;
                    startingPos = pos + 1;
                }
            }
        }

        if(!isInTri && pos - 1 >= 0) {
            isInTri = checkPointInTriangle(pos - 1, tri, encIdx, triPoints,
                                           query, &s, &t, eps, true);
            if(isInTri) {
                trianglePos = encIdx[pos - 1];
                stoppedAt = 0;
            } else {
                double centerDist;
                double dist = computePointTriDist(
                    encIdx[pos - 1], query, centers, tri, triPoints,
                    &centerDist);
                if(dist < minDist) {
                    minDist = dist;
                    startingPos = pos - 1;
                } else if(dist == minDist && centerDist < minCenterDist) {
                    minDist = dist;
                    minCenterDist = centerDist;
                    startingPos = pos - 1;
                }
            }
        }

        if(!isInTri && nTri > 1) {
            // Find the nearest opposite triangle to the query point from
            // the nearest center found.
            TriOpp nearest = triOpp[encIdx[startingPos]];
            int maxSkips = 9;

            int nextNearest = encIdx[startingPos];
            int visited[10] = {nextNearest, -1, -1, -1, -1, -1,
                               -1, -1, -1, -1};

            while(off < maxSkips && !isInTri) {
                int prevNearest = nextNearest;
                nextNearest = -1;

                minDist = CUDART_INF;
                minCenterDist = CUDART_INF;
                for(int i = 0; i < DEG && !isInTri; i++) {
                    int oppTriIdx = nearest.getOppTri(i);

                    if(oppTriIdx == -1) {
                        continue;
                    }

                    if(visitedAlready(oppTriIdx, visited, off + 1)) {
                        continue;
                    }

                    isInTri = checkPointInTriangle(
                        oppTriIdx, tri, NULL, triPoints, query, &s, &t,
                        eps, false);

                    if(!isInTri) {
                        double centerDist;
                        double dist = computePointTriDist(
                            oppTriIdx, query, centers, tri, triPoints,
                            &centerDist);

                        if(dist < minDist) {
                            minDist = dist;
                            nextNearest = oppTriIdx;
                        } else if(dist == minDist &&
                                  centerDist < minCenterDist) {
                            minDist = dist;
                            minCenterDist = centerDist;
                            nextNearest = oppTriIdx;
                        }
                    } else {
                        trianglePos = oppTriIdx;
                        stoppedAt = off;
                    }
                }

                if(!isInTri) {
                    nearest = triOpp[nextNearest];
                    visited[off + 1] = nextNearest;
                }
                off++;
            }
        }

        /**if(debug != NULL) {
            debug[3 * idx] = pos;
            debug[3 * idx + 1] = stoppedAt;
            debug[3 * idx + 2] = off;
        }**/

        out[idx] = trianglePos;
        if(findCoords) {
            RealType* outCoords = coords + 3 * idx;
            outCoords[0] = 1 - s - t;
            outCoords[1] = s;
            outCoords[2] = t;
        }
    }
}
"""

DELAUNAY_MODULE = cupy.RawModule(
    code=KERNEL_DIVISION, options=('-std=c++11', '-w',),
    name_expressions=['kerMakeFirstTri', 'kerInitPointLocationFast',
                      'kerInitPointLocationExact', 'kerVoteForPoint',
                      'kerPickWinnerPoint', 'kerShiftValues<Tri>',
                      'kerShiftValues<char>', 'kerShiftValues<int>',
                      'kerShiftOpp', 'kerShiftTriIdx',
                      'kerSplitPointsFast', 'kerSplitPointsExactSoS',
                      'kerSplitTri', 'isTriActive', 'kerMarkSpecialTris',
                      'kerCheckDelaunayFast', 'kerCheckDelaunayExact_Fast',
                      'kerCheckDelaunayExact_Exact', 'kerMarkRejectedFlips',
                      'kerFlip', 'kerUpdateOpp', 'kerUpdateFlipTrace',
                      'kerRelocatePointsFast', 'kerRelocatePointsExact',
                      'kerMarkInfinityTri', 'kerCollectFreeSlots',
                      'kerMakeCompactMap', 'kerCompactTris',
                      'kerUpdateVertIdx', 'getMortonNumber',
                      'computeDistance2D', 'kerInitPredicate',
                      'makeKeyFromTriHasVert', 'kerCheckIfCoplanarPoints',
                      'kerEncodeEdges', 'kerEncBarycenters',
                      'kerFindClosestTri', 'kerCountVertexNeighbors',
                      'kerFillVertexNeighbors'])


N_BLOCKS = 512
BLOCK_SZ = 128

PRED_N_BLOCKS = 64
PRED_BLOCK_SZ = 32


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
    ker_init_point_loc_fast((PRED_N_BLOCKS,), (PRED_BLOCK_SZ,), (
        vert_tri, n_vert, exact_check, counter, tri, inf_idx,
        point_vec, points_idx, pred_consts))


def vote_for_point(vert_tri, n_vert, tri, vert_circle, tri_circle,
                   no_sample, inf_idx, points, pred_consts):
    ker_vote_for_point = DELAUNAY_MODULE.get_function('kerVoteForPoint')
    ker_vote_for_point((N_BLOCKS,), (BLOCK_SZ,), (
        vert_tri, n_vert, tri, vert_circle, tri_circle,
        no_sample, inf_idx, points, pred_consts
    ))


def pick_winner_point(vert_tri, n_vert, vert_circle, tri_circle,
                      tri_to_vert, no_sample):
    ker_pick_winner_point = DELAUNAY_MODULE.get_function('kerPickWinnerPoint')
    ker_pick_winner_point((N_BLOCKS,), (BLOCK_SZ,), (
        vert_tri, n_vert, vert_circle, tri_circle, tri_to_vert, no_sample))


def shift(shift_idx, in_arr, out_arr, type_str):
    ker_shift = DELAUNAY_MODULE.get_function(f'kerShiftValues<{type_str}>')
    ker_shift((N_BLOCKS,), (BLOCK_SZ,), (
        shift_idx, int(shift_idx.shape[0]), in_arr, out_arr))


def shift_opp_tri(shift_idx, in_arr, out_arr):
    ker_shift = DELAUNAY_MODULE.get_function('kerShiftOpp')
    ker_shift((N_BLOCKS,), (BLOCK_SZ,), (
        shift_idx, int(shift_idx.shape[0]), in_arr, out_arr))


def shift_tri_idx(idx, shift_idx):
    ker_shift = DELAUNAY_MODULE.get_function('kerShiftTriIdx')
    ker_shift((N_BLOCKS,), (BLOCK_SZ,), (
        idx, int(idx.shape[0]), shift_idx))


def split_points_fast(vert_tri, tri_to_vert, tri, ins_tri_map,
                      exact_check, counter, tri_num, ins_tri_num,
                      inf_idx, points, points_idx, pred_consts):
    ker_split_points_fast = DELAUNAY_MODULE.get_function('kerSplitPointsFast')
    ker_split_points_fast((N_BLOCKS,), (BLOCK_SZ,), (
        vert_tri, int(vert_tri.shape[0]), tri_to_vert, tri, ins_tri_map,
        exact_check, counter, int(tri_num), int(ins_tri_num), inf_idx, points,
        points_idx, pred_consts))


def split_points_exact(vert_tri, tri_to_vert, tri, ins_tri_map,
                       exact_check, counter, tri_num, ins_tri_num,
                       inf_idx, points, points_idx, pred_consts):
    ker_split_points_exact = DELAUNAY_MODULE.get_function(
        'kerSplitPointsExactSoS')
    ker_split_points_exact((PRED_BLOCK_SZ,), (PRED_N_BLOCKS,), (
        vert_tri, tri_to_vert, tri, ins_tri_map,
        exact_check, counter, int(tri_num), int(ins_tri_num), inf_idx, points,
        points_idx, pred_consts))


def split_tri(split_tri, tri, tri_opp, tri_info, ins_tri_map, tri_to_vert,
              tri_num, ins_tri_num):
    ker_split_tri = DELAUNAY_MODULE.get_function('kerSplitTri')
    ker_split_tri((N_BLOCKS,), (32,), (
        split_tri, int(split_tri.shape[0]), tri, tri_opp, tri_info,
        ins_tri_map, tri_to_vert, int(tri_num), int(ins_tri_num)))


def is_tri_active(tri_info):
    out = cupy.empty_like(tri_info, dtype=cupy.bool_)
    ker_is_tri_active = DELAUNAY_MODULE.get_function('isTriActive')
    ker_is_tri_active((N_BLOCKS,), (BLOCK_SZ,), (
        tri_info, out, tri_info.shape[0]))
    return out


def mark_special_tris(tri_info, tri_opp):
    ker_mark_special_tris = DELAUNAY_MODULE.get_function('kerMarkSpecialTris')
    ker_mark_special_tris((N_BLOCKS,), (BLOCK_SZ,), (
        tri_info, tri_opp, tri_info.shape[0]))


def check_delaunay_fast(act_tri, tri, tri_opp, tri_info,
                        tri_vote, act_tri_num, inf_idx,
                        points, pred_consts):
    ker_check_delaunay_fast = DELAUNAY_MODULE.get_function(
        'kerCheckDelaunayFast')
    ker_check_delaunay_fast((N_BLOCKS,), (BLOCK_SZ,), (
        act_tri, tri, tri_opp, tri_info, tri_vote, act_tri_num,
        inf_idx, points, pred_consts))


def check_delaunay_exact_fast(act_tri, tri, tri_opp, tri_info,
                              tri_vote, exact_check_vi, act_tri_num,
                              counter, inf_idx, points, pred_consts):
    ker_check_delaunay_e_f = DELAUNAY_MODULE.get_function(
        'kerCheckDelaunayExact_Fast')
    ker_check_delaunay_e_f((N_BLOCKS,), (BLOCK_SZ,), (
        act_tri, tri, tri_opp, tri_info, tri_vote, exact_check_vi, act_tri_num,
        counter, inf_idx, points, pred_consts))


def check_delaunay_exact_exact(tri, tri_opp, tri_vote, exact_check_vi,
                               counter, inf_idx, points, org_point_idx,
                               pred_consts):
    ker_check_delaunay_e_e = DELAUNAY_MODULE.get_function(
        'kerCheckDelaunayExact_Exact')

    ker_check_delaunay_e_e((PRED_N_BLOCKS,), (PRED_BLOCK_SZ,), (
        tri, tri_opp, tri_vote, exact_check_vi, counter, inf_idx,
        points, org_point_idx, pred_consts))


def mark_rejected_flips(act_tri, tri_opp, tri_vote, tri_info,
                        flip_to_tri, act_tri_num):
    ker_mark_rejected_flips = DELAUNAY_MODULE.get_function(
        'kerMarkRejectedFlips')
    ker_mark_rejected_flips((N_BLOCKS,), (BLOCK_SZ,), (
        act_tri, tri_opp, tri_vote, tri_info, flip_to_tri, act_tri_num))


def flip(flip_to_tri, tri, tri_opp, tri_info, tri_msg, act_tri, flip_arr,
         org_flip_num, act_tri_num, mode):
    ker_flip = DELAUNAY_MODULE.get_function('kerFlip')
    ker_flip((N_BLOCKS,), (32,), (
        flip_to_tri, flip_to_tri.shape[0], tri, tri_opp, tri_info,
        tri_msg, act_tri, flip_arr, org_flip_num, act_tri_num, mode))


def update_opp(flip_vec, tri_opp, tri_msg, flip_to_tri,
               org_flip_num, flip_num):
    ker_update_opp = DELAUNAY_MODULE.get_function('kerUpdateOpp')
    ker_update_opp((BLOCK_SZ,), (32,), (
        flip_vec, tri_opp, tri_msg, flip_to_tri, org_flip_num, flip_num))


def update_flip_trace(flip_arr, tri_to_flip, org_flip_num, flip_num):
    ker_update_flip_trace = DELAUNAY_MODULE.get_function('kerUpdateFlipTrace')
    ker_update_flip_trace((N_BLOCKS,), (BLOCK_SZ,), (
        flip_arr, tri_to_flip, org_flip_num, flip_num))


def relocate_points_fast(vert_tri, tri_to_flip, flip_arr, exact_check,
                         counter, inf_idx, points, org_point_idx, pred_consts):
    ker_relocate_points_fast = DELAUNAY_MODULE.get_function(
        'kerRelocatePointsFast')
    ker_relocate_points_fast((N_BLOCKS,), (BLOCK_SZ,), (
        vert_tri, vert_tri.shape[0], tri_to_flip, flip_arr, exact_check,
        counter, inf_idx, points, org_point_idx, pred_consts))


def relocate_points_exact(vert_tri, tri_to_flip, flip_arr, exact_check,
                          counter, inf_idx, points, org_point_idx,
                          pred_consts):
    ker_relocate_points_exact = DELAUNAY_MODULE.get_function(
        'kerRelocatePointsExact')
    ker_relocate_points_exact((N_BLOCKS,), (BLOCK_SZ,), (
        vert_tri, tri_to_flip, flip_arr, exact_check, counter, inf_idx,
        points, org_point_idx, pred_consts))


def mark_inf_tri(tri, tri_info, opp, inf_idx):
    ker_mark_inf_tri = DELAUNAY_MODULE.get_function('kerMarkInfinityTri')
    ker_mark_inf_tri((N_BLOCKS,), (BLOCK_SZ,), (
        tri, tri.shape[0], tri_info, opp, inf_idx))


def collect_free_slots(tri_info, prefix, free_arr, new_tri_num):
    ker_collect_free = DELAUNAY_MODULE.get_function('kerCollectFreeSlots')
    ker_collect_free((N_BLOCKS,), (BLOCK_SZ,), (
        tri_info, prefix, free_arr, new_tri_num))


def make_compact_map(tri_info, prefix, free_arr, new_tri_num):
    ker_make_compact = DELAUNAY_MODULE.get_function('kerMakeCompactMap')
    ker_make_compact((N_BLOCKS,), (BLOCK_SZ,), (
        tri_info, tri_info.shape[0], prefix, free_arr, new_tri_num))


def compact_tris(tri_info, prefix, tri, tri_opp, new_tri_num):
    ker_compact_tris = DELAUNAY_MODULE.get_function('kerCompactTris')
    ker_compact_tris((N_BLOCKS,), (BLOCK_SZ,), (
        tri_info, tri_info.shape[0], prefix, tri, tri_opp, new_tri_num))


def update_vert_idx(tri, tri_info, org_point_idx):
    ker_map_tri = DELAUNAY_MODULE.get_function('kerUpdateVertIdx')
    ker_map_tri((N_BLOCKS,), (BLOCK_SZ,), (
        tri, tri.shape[0], tri_info, org_point_idx))


def get_morton_number(points, n_points, min_val, range_val, values):
    ker_morton = DELAUNAY_MODULE.get_function('getMortonNumber')
    ker_morton((N_BLOCKS,), (BLOCK_SZ,), (
        points, n_points, min_val, range_val, values))


def compute_distance_2d(points, v0, v1, values):
    ker_dist = DELAUNAY_MODULE.get_function('computeDistance2D')
    ker_dist((N_BLOCKS,), (BLOCK_SZ,), (
        points, points.shape[0], v0, v1, values))


def init_predicate(pred_values):
    ker_init_predicate = DELAUNAY_MODULE.get_function('kerInitPredicate')
    ker_init_predicate((1,), (1,), (pred_values))


def make_key_from_tri_has_vert(tri_has_vert, out):
    ker_key = DELAUNAY_MODULE.get_function('makeKeyFromTriHasVert')
    ker_key((N_BLOCKS,), (BLOCK_SZ,), (
        tri_has_vert, tri_has_vert.shape[0], out))


def check_if_coplanar_points(points, pa_idx, pb_idx, pc_idx, det):
    ker_ori = DELAUNAY_MODULE.get_function('kerCheckIfCoplanarPoints')
    ker_ori((1,), (1,), (points, pa_idx, pb_idx, pc_idx, det))


def encode_edges(tri, tri_points, min_val, range_val, out, edges):
    ker_enc_edges = DELAUNAY_MODULE.get_function('kerEncodeEdges')
    ker_enc_edges((N_BLOCKS,), (BLOCK_SZ,), (
        tri, tri.shape[0], tri_points, min_val, range_val, out, edges))


def encode_barycenters(tri, tri_points, min_val, range_val, out, centers):
    ker_enc = DELAUNAY_MODULE.get_function('kerEncBarycenters')
    ker_enc((N_BLOCKS,), (BLOCK_SZ,), (
        tri, tri.shape[0], tri_points, min_val, range_val, out, centers))


def find_closest_tri(queries, tri, tri_opp, enc_idx, tri_enc, tri_points,
                     centers, min_val, range_val, min_axis, max_axis,
                     eps, find_coords, out, coords):
    ker_find = DELAUNAY_MODULE.get_function('kerFindClosestTri')
    ker_find((N_BLOCKS,), (BLOCK_SZ,), (
        queries, queries.shape[0], tri, tri.shape[0], tri_opp,
        enc_idx, tri_enc, tri_points, centers, min_val, range_val,
        min_axis, max_axis, eps, find_coords, out, coords
    ))


def count_vertex_neighbors(edges, vertex_count):
    ker_count = DELAUNAY_MODULE.get_function('kerCountVertexNeighbors')
    ker_count((N_BLOCKS,), (BLOCK_SZ,),
              (edges, edges.shape[0], vertex_count, vertex_count.shape[0]))


def fill_vertex_neighbors(edges, vertex_off, vertex_count, vertex_neighbors):
    ker_fill = DELAUNAY_MODULE.get_function('kerFillVertexNeighbors')
    ker_fill((N_BLOCKS,), (BLOCK_SZ,), (
        edges, edges.shape[0], vertex_off, vertex_count, vertex_neighbors))
