#ifndef SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_MULTIPOLE_H__
#define SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_MULTIPOLE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #include <stdbool.h>
    #include <stdint.h>
    #include <stdlib.h>
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/_impl/definitions.h"
    #include "sixtracklib/common/impl/buffer_type.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ========================================================================= */

typedef SIXTRL_INT64_T NS(multipole_order_t);
typedef SIXTRL_REAL_T  NS(multipole_real_t);

typedef struct NS(MultiPole)
{
    NS(multipole_order_t) order   SIXTRL_ALIGN( 8 );
    NS(multipole_real_t)  length  SIXTRL_ALIGN( 8 );
    NS(multipole_real_t)  hxl     SIXTRL_ALIGN( 8 );
    NS(multipole_real_t)  hyl     SIXTRL_ALIGN( 8 );

    SIXTRL_DATAPTR_DEC NS(multipole_real_t)*
        SIXTRL_RESTRICT bal SIXTRL_ALIGN( 8 );
}
NS(MultiPole);

/* ========================================================================= */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(MultiPole_get_num_dataptrs)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC NS(MultiPole)* NS(MultiPole_preset)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC NS(multipole_real_t) NS(MultiPole_get_length)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC NS(multipole_real_t) NS(MultiPole_get_hxl)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC NS(multipole_real_t) NS(MultiPole_get_hyl)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC NS(multipole_order_t) NS(MultiPole_get_order)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(MultiPole_get_bal_size)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_DATAPTR_DEC NS(multipole_real_t) const*
NS(MultiPole_get_const_bal)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_DATAPTR_DEC NS(multipole_real_t)* NS(MultiPole_get_bal)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole );

SIXTRL_FN SIXTRL_DATAPTR_DEC NS(multipole_real_t) NS(MultiPole_get_bal_value)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index );

SIXTRL_FN SIXTRL_DATAPTR_DEC NS(multipole_real_t) NS(MultiPole_get_knl_value)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index );

SIXTRL_FN SIXTRL_DATAPTR_DEC NS(multipole_real_t) NS(MultiPole_get_ksl_value)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_length)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const length );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_hxl)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const length );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_hyl)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const length );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_order)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const order );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_bal)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    SIXTRL_ARGPTR_DEC NS(multipole_real_t) const* SIXTRL_RESTRICT bal );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_knl)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    SIXTRL_ARGPTR_DEC NS(multipole_real_t) const* SIXTRL_RESTRICT knl );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_ksl)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    SIXTRL_ARGPTR_DEC NS(multipole_real_t) const* SIXTRL_RESTRICT ksl );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_assign_bal)(
    SIXTRL_ARGPTR_DEC  NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const order,
    SIXTRL_DATAPTR_DEC NS(multipole_real_t)* SIXTRL_RESTRICT bal );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_bal_value)(
    SIXTRL_ARGPTR_DEC  NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index, NS(multipole_real_t) const bal );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_knl_value)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index,
    NS(multipole_real_t) const knl_i );

SIXTRL_FN SIXTRL_STATIC void NS(MultiPole_set_ksl_value)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index,
    NS(multipole_real_t) const ksl_i );


SIXTRL_FN SIXTRL_STATIC bool NS(MultiPole_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(MultiPole)* NS(MultiPole_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order );

SIXTRL_FN SIXTRL_STATIC SIXTRL_ARGPTR_DEC NS(MultiPole)* NS(MultiPole_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order,
    SIXTRL_DATAPTR_DEC NS(multipole_real_t) const* SIXTRL_RESTRICT bal,
    NS(multipole_real_t)  const length,
    NS(multipole_real_t)  const hxl,
    NS(multipole_real_t)  const hyl );

/* ========================================================================= */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

/* ************************************************************************* */
/* *****          Implementation of C inline functions                   *** */
/* ************************************************************************* */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/buffer.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if !defined(  _GPUCODE ) && defined( __cplusplus )
extern "C" {
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

SIXTRL_FN SIXTRL_STATIC NS(buffer_size_t) NS(MultiPole_get_num_dataptrs)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    typedef NS(multipole_order_t)   mp_order_t;
    typedef NS(buffer_size_t)       buf_size_t;

    return ( ( multipole != SIXTRL_NULLPTR ) &&
             ( multipole->order >= ( mp_order_t )0 ) )
        ? ( buf_size_t )1u : ( buf_size_t )0u;
}

SIXTRL_FN SIXTRL_STATIC NS(multipole_order_t) NS(_calculate_factorial)(
    NS(multipole_order_t) const n )
{
    NS(multipole_order_t) result = 1;
    NS(multipole_order_t) ii     = 1;

    for( ; ii <= n ; ++ii )
    {
        result *= ii;
    }

    return result;
}

SIXTRL_INLINE NS(MultiPole)* NS(MultiPole_preset)(
    NS(MultiPole)* SIXTRL_RESTRICT multipole )
{
    if( multipole != SIXTRL_NULLPTR )
    {
        typedef NS(multipole_order_t) mp_order_t;
        SIXTRL_STATIC_VAR mp_order_t DEFAULT_ORDER = ( mp_order_t )-1;

        NS(MultiPole_set_length)( multipole, ( NS(multipole_real_t) )0 );
        NS(MultiPole_set_hxl)(    multipole, ( NS(multipole_real_t) )0 );
        NS(MultiPole_set_hyl)(    multipole, ( NS(multipole_real_t) )0 );
        NS(MultiPole_assign_bal)( multipole, DEFAULT_ORDER, SIXTRL_NULLPTR );
    }

    return multipole;
}

SIXTRL_INLINE NS(multipole_real_t) NS(MultiPole_get_length)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return ( multipole != SIXTRL_NULLPTR )
        ? multipole->length : ( NS(multipole_real_t) )0;
}

SIXTRL_INLINE NS(multipole_real_t) NS(MultiPole_get_hxl)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return ( multipole != SIXTRL_NULLPTR )
        ? multipole->hxl : ( NS(multipole_real_t) )0;
}

SIXTRL_INLINE NS(multipole_real_t) NS(MultiPole_get_hyl)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return ( multipole != SIXTRL_NULLPTR )
        ? multipole->hyl : ( NS(multipole_real_t) )0;
}

SIXTRL_INLINE NS(multipole_order_t) NS(MultiPole_get_order)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return ( multipole != SIXTRL_NULLPTR )
        ? multipole->order : ( SIXTRL_UINT64_T )0u;
}

SIXTRL_INLINE SIXTRL_DATAPTR_DEC NS(multipole_real_t) const*
NS(MultiPole_get_const_bal)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    SIXTRL_ASSERT(
        ( multipole == SIXTRL_NULLPTR ) ||
        ( ( ( NS(MultiPole_get_order)( multipole ) >= 0 ) &&
            ( multipole->bal != SIXTRL_NULLPTR ) ) ||
          ( ( NS(MultiPole_get_order)( multipole ) < 0 ) &&
            ( multipole->bal == SIXTRL_NULLPTR ) ) ) );

    return ( multipole != SIXTRL_NULLPTR ) ? multipole->bal : SIXTRL_NULLPTR;
}

SIXTRL_INLINE SIXTRL_DATAPTR_DEC NS(multipole_real_t)* NS(MultiPole_get_bal)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole )
{
    return ( NS(multipole_real_t)* )NS(MultiPole_get_const_bal)( multipole );
}

SIXTRL_INLINE NS(multipole_real_t) NS(MultiPole_get_bal_value)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index )
{
    typedef NS(buffer_size_t) buf_size_t;
    typedef SIXTRL_DATAPTR_DEC NS(multipole_real_t) const* ptr_to_bal_t;

    buf_size_t const bal_size = NS(MultiPole_get_bal_size)( multipole );
    ptr_to_bal_t bal = NS(MultiPole_get_const_bal)( multipole );

    return ( ( bal != SIXTRL_NULLPTR ) && ( bal_size > index ) )
        ? bal[ index ] : ( NS(multipole_real_t) )0.0;
}

SIXTRL_INLINE NS(buffer_size_t) NS(MultiPole_get_bal_size)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole )
{
    return ( multipole != SIXTRL_NULLPTR )
        ?  ( ( multipole->order >= ( NS(multipole_order_t ) )0 )
                ? ( ( NS(buffer_size_t) )( 2 * multipole->order + 1 ) )
                : ( ( NS(buffer_size_t) )0u ) )
        :  ( ( NS(buffer_size_t) )0u );
}


SIXTRL_INLINE NS(multipole_real_t) NS(MultiPole_get_knl_value)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t const bal_index = ( buf_size_t )2u * index;

    return NS(MultiPole_get_bal_value)( multipole, bal_index ) *
           NS(_calculate_factorial)( index );
}

SIXTRL_INLINE NS(multipole_real_t) NS(MultiPole_get_ksl_value)(
    SIXTRL_ARGPTR_DEC const NS(MultiPole) *const SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index )
{
    typedef NS(buffer_size_t) buf_size_t;
    buf_size_t const bal_index = ( buf_size_t )2u * index + ( buf_size_t )1u;

    return NS(MultiPole_get_bal_value)( multipole, bal_index ) *
           NS(_calculate_factorial)( index );
}

SIXTRL_INLINE void NS(MultiPole_set_length)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const length )
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    multipole->length = length;
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_hxl)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const hxl )
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    multipole->hxl = hxl;
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_hyl)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_real_t) const hyl )
{
    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );
    multipole->hyl = hyl;
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_order)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const order )
{
    SIXTRL_STATIC_VAR NS(multipole_order_t) const
        INV_ORDER = ( NS(multipole_order_t) )-1;

    SIXTRL_ASSERT( ( multipole != SIXTRL_NULLPTR ) && ( order >= INV_ORDER ) );
    multipole->order = order;
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_bal)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    SIXTRL_ARGPTR_DEC NS(multipole_real_t) const* SIXTRL_RESTRICT bal )
{
    NS(buffer_size_t) const bal_size = NS(MultiPole_get_bal_size)( multipole );

    SIXTRL_DATAPTR_DEC NS(multipole_real_t)* dest =
        NS(MultiPole_get_bal)( multipole );

    SIXTRACKLIB_COPY_VALUES( NS(multipole_real_t), dest, bal, bal_size );

    return;
}

SIXTRL_INLINE void NS(MultiPole_set_knl)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    SIXTRL_ARGPTR_DEC NS(multipole_real_t) const* SIXTRL_RESTRICT knl )
{
    NS(multipole_order_t) const order = NS(MultiPole_get_order)( multipole );

    SIXTRL_ARGPTR_DEC NS(multipole_real_t)* bal =
        NS(MultiPole_get_bal)( multipole );

    if( ( order >= 0 ) &&
        ( bal != SIXTRL_NULLPTR ) &&
        ( knl != SIXTRL_NULLPTR ) )
    {
        bal[ 0 ] = knl[ 0 ];

        if( order > 0 )
        {
            NS(multipole_order_t)    ii = 1;
            NS(buffer_size_t) jj = ( NS(buffer_size_t) )2u;
            NS(multipole_real_t)   fact = ( NS(multipole_real_t) )1;

            for( ; ii <= order ; jj += 2u,
                    fact *= ( NS(multipole_real_t) )ii++ )
            {
                SIXTRL_ASSERT( fact > ( NS(multipole_real_t) )0 );
                bal[ jj ] = knl[ ii ] / fact;
            }
        }
    }

    return;
}

SIXTRL_INLINE void NS(MultiPole_set_ksl)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    SIXTRL_ARGPTR_DEC NS(multipole_real_t) const* SIXTRL_RESTRICT ksl )
{
    NS(multipole_order_t) const order = NS(MultiPole_get_order)( multipole );

    SIXTRL_ARGPTR_DEC NS(multipole_real_t)* bal =
        NS(MultiPole_get_bal)( multipole );

    if( ( order >= 0 ) &&
        ( bal != SIXTRL_NULLPTR ) && ( ksl != SIXTRL_NULLPTR ) )
    {
        bal[ 1 ] = ksl[ 0 ];

        if( order > 0 )
        {
            NS(multipole_order_t)    ii = 1;
            NS(buffer_size_t) jj = ( NS(buffer_size_t) )3u;
            NS(multipole_real_t)   fact = ( NS(multipole_real_t) )1;

            for( ; ii <= order ; jj += 2u,
                    fact *= ( NS(multipole_real_t) )ii++ )
            {
                SIXTRL_ASSERT( fact > ( NS(multipole_real_t) )0 );
                bal[ jj ] = ksl[ ii ] / fact;
            }
        }
    }

    return;
}

SIXTRL_INLINE void NS(MultiPole_assign_bal)(
    SIXTRL_ARGPTR_DEC  NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(multipole_order_t) const order,
    SIXTRL_DATAPTR_DEC NS(multipole_real_t)* SIXTRL_RESTRICT bal_ptr )
{
    typedef NS(multipole_order_t) mp_order_t;
    SIXTRL_STATIC_VAR mp_order_t const ZERO_ORDER = ( mp_order_t )0;

    SIXTRL_ASSERT( multipole != SIXTRL_NULLPTR );

    if( ( multipole != SIXTRL_NULLPTR ) &&
        ( ( ( order <  ZERO_ORDER ) && ( bal_ptr == SIXTRL_NULLPTR ) ) ||
          ( ( order >= ZERO_ORDER ) && ( bal_ptr != SIXTRL_NULLPTR ) ) ) )
    {
        multipole->order = order;
        multipole->bal   = bal_ptr;
    }

    return;
}

SIXTRL_INLINE void NS(MultiPole_set_bal_value)(
    SIXTRL_ARGPTR_DEC NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index, NS(multipole_real_t) const bal_value )
{
    typedef NS(multipole_real_t)            mp_real_t;
    typedef NS(buffer_size_t)               buf_size_t;
    typedef SIXTRL_DATAPTR_DEC mp_real_t*   ptr_to_bal_t;

    ptr_to_bal_t bal          = NS(MultiPole_get_bal)( multipole );
    buf_size_t const bal_size = NS(MultiPole_get_bal_size)( multipole );

    if( ( bal_size > index ) && ( bal != SIXTRL_NULLPTR ) )
    {
        bal[ index ] = bal_value;
    }

    return;
}

SIXTRL_INLINE void NS(MultiPole_set_knl_value)(
    SIXTRL_ARGPTR_DEC  NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index,
    NS(multipole_real_t)  const knl_value )
{
    NS(multipole_real_t) const bal_value = knl_value / (
        ( NS(multipole_real_t) )NS(_calculate_factorial)( index ) );

    NS(MultiPole_set_bal_value)( multipole, 2u * index, bal_value );
    return;
}

SIXTRL_INLINE void NS(MultiPole_set_ksl_value)(
    SIXTRL_ARGPTR_DEC  NS(MultiPole)* SIXTRL_RESTRICT multipole,
    NS(buffer_size_t) const index,
    NS(multipole_real_t) const ksl_value )
{
    NS(multipole_real_t) const bal_value = ksl_value / (
        ( NS(multipole_real_t) )NS(_calculate_factorial)( index ) );

    NS(MultiPole_set_bal_value)( multipole, 2u * index + 1u, bal_value );
    return;
}

SIXTRL_INLINE bool NS(MultiPole_can_be_added)(
    SIXTRL_ARGPTR_DEC const NS(Buffer) *const SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_objects,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_slots,
    SIXTRL_ARGPTR_DEC NS(buffer_size_t)* SIXTRL_RESTRICT ptr_requ_dataptrs )
{
    typedef NS(buffer_size_t) buf_size_t;

    buf_size_t const bal_size = ( buf_size_t )2u + ( ( order >= 0 )
        ? ( buf_size_t )2u * order : ( buf_size_t )0u );

    buf_size_t const num_dataptrs = ( buf_size_t )1u;
    buf_size_t const sizes[]  = { sizeof( NS(multipole_real_t) ) };
    buf_size_t const counts[] = { bal_size };

    return NS(Buffer_can_add_object)( buffer, sizeof( NS(MultiPole) ),
            num_dataptrs, sizes, counts, ptr_requ_objects, ptr_requ_slots,
                ptr_requ_dataptrs );
}


SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(MultiPole)* NS(MultiPole_new)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order )
{
    typedef NS(MultiPole)                   elem_t;
    typedef NS(buffer_size_t)               buf_size_t;
    typedef SIXTRL_ARGPTR_DEC elem_t*       ptr_to_elem_t;
    typedef SIXTRL_ARGPTR_DEC NS(Object)*   ptr_to_obj_t;

    ptr_to_obj_t ptr_obj = SIXTRL_NULLPTR;
    buf_size_t const obj_size = sizeof( elem_t );

    NS(object_type_id_t) const type_id = NS(OBJECT_TYPE_MULTIPOLE);

    elem_t temp_obj;
    NS(MultiPole_preset)( &temp_obj );
    NS(MultiPole_set_order)( &temp_obj, order );

    if( order >= ( NS(multipole_order_t) )0 )
    {
        buf_size_t const bal_size     = ( buf_size_t )( 2 * order + 1 );
        buf_size_t const num_dataptrs = ( buf_size_t )1u;

        SIXTRL_ARGPTR_DEC buf_size_t const offsets[] =
        {
            offsetof( elem_t, bal )
        };

        SIXTRL_ARGPTR_DEC buf_size_t const sizes[] =
        {
            sizeof( NS(multipole_real_t) )
        };

        SIXTRL_ARGPTR_DEC buf_size_t const counts[] = { bal_size };
        SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

        ptr_obj = NS(Buffer_add_object)( buffer, &temp_obj, obj_size , type_id,
                num_dataptrs, offsets, sizes, counts );
    }
    else
    {
        buf_size_t const num_dataptrs = ( buf_size_t )0u;

        ptr_obj = NS(Buffer_add_object)( buffer, &temp_obj, obj_size , type_id,
                num_dataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR );
    }

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)( ptr_obj );
}

SIXTRL_INLINE SIXTRL_ARGPTR_DEC NS(MultiPole)* NS(MultiPole_add)(
    SIXTRL_ARGPTR_DEC NS(Buffer)* SIXTRL_RESTRICT buffer,
    NS(multipole_order_t) const order,
    SIXTRL_DATAPTR_DEC NS(multipole_real_t) const* SIXTRL_RESTRICT ptr_to_bal,
    NS(multipole_real_t)  const length,
    NS(multipole_real_t)  const hxl,
    NS(multipole_real_t)  const hyl )
{
    typedef NS(MultiPole)                               elem_t;
    typedef NS(buffer_size_t)                           buf_size_t;
    typedef SIXTRL_ARGPTR_DEC elem_t*                   ptr_to_elem_t;
    typedef SIXTRL_ARGPTR_DEC NS(Object)*               ptr_to_obj_t;
    typedef SIXTRL_ARGPTR_DEC NS(multipole_real_t )*    ptr_to_bal_t;

    SIXTRL_STATIC_VAR NS(multipole_order_t) const INV_ORDER =
        ( NS(multipole_order_t) )-1;

    ptr_to_obj_t ptr_obj = SIXTRL_NULLPTR;
    buf_size_t const obj_size = sizeof( elem_t );

    NS(object_type_id_t) const type_id = NS(OBJECT_TYPE_MULTIPOLE);

    elem_t temp_obj;
    NS(MultiPole_set_length)( &temp_obj, length );
    NS(MultiPole_set_hxl)(    &temp_obj, hxl );
    NS(MultiPole_set_hyl)(    &temp_obj, hyl );

    if( order > INV_ORDER )
    {
        buf_size_t const num_dataptrs = ( buf_size_t )1u;

        SIXTRL_ARGPTR_DEC buf_size_t const offsets[] =
        {
            offsetof( elem_t, bal )
        };

        SIXTRL_ARGPTR_DEC buf_size_t const sizes[] =
        {
            sizeof( NS(multipole_real_t) )
        };

        SIXTRL_ARGPTR_DEC buf_size_t const counts[] =
        {
            ( buf_size_t )( 2 * order + 1 )
        };

        if( ptr_to_bal != SIXTRL_NULLPTR )
        {
            NS(MultiPole_assign_bal)(
                &temp_obj, order, ( ptr_to_bal_t )ptr_to_bal );
        }
        else
        {
            NS(MultiPole_assign_bal)( &temp_obj, INV_ORDER, SIXTRL_NULLPTR );
            NS(MultiPole_set_order)( &temp_obj, order );
        }

        SIXTRL_ASSERT( num_dataptrs == ( buf_size_t )1u );

        ptr_obj = NS(Buffer_add_object)( buffer, &temp_obj, obj_size , type_id,
            num_dataptrs, offsets, sizes, counts );
    }
    else
    {
        buf_size_t const num_dataptrs = ( buf_size_t )0u;
        NS(MultiPole_assign_bal)( &temp_obj, INV_ORDER, SIXTRL_NULLPTR );

        ptr_obj = NS(Buffer_add_object)( buffer, &temp_obj, obj_size, type_id,
            num_dataptrs, SIXTRL_NULLPTR, SIXTRL_NULLPTR, SIXTRL_NULLPTR );
    }

    SIXTRL_ASSERT( NS(MultiPole_get_order)( &temp_obj ) == order );

    return ( ptr_to_elem_t )( uintptr_t )NS(Object_get_begin_addr)( ptr_obj );
}

#if !defined(  _GPUCODE ) && defined( __cplusplus )
}
#endif /* !defined(  _GPUCODE ) && defined( __cplusplus ) */

#endif /* SIXTRACKLIB_COMMON_IMPL_BEAM_ELEMENT_MULTIPOLE_H__ */
/* end: sixtracklib/common/impl/be_multipole.h  */
