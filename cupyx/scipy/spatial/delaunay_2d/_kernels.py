
import cupy


KERNEL_DIVISION = r"""

#define INLINE_H_D __forceinline__ __device__

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

__global__ void kerMakeFirstTri
(
Tri*	triArr,
TriOpp*	oppArr,
char*	triInfoArr,
Tri     tri,
int     infIdx
)
{
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
"""

DELAUNAY_MODULE = cupy.RawModule(code=KERNEL_DIVISION, options=('-std=c++11',),
                                 name_expressions=['kerMakeFirstTri'])
