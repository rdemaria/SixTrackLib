#ifndef SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_BASE_H__
#define SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_BASE_H__

#if !defined( SIXTRL_NO_SYSTEM_INCLUDES )
    #if defined( __cplusplus )
        #include <algorithm>
        #include <cstddef>
        #include <cstdint>
        #include <cstdlib>
        #include <map>
        #include <memory>
        #include <string>
        #include <vector>
    #else /* !defined( __cplusplus ) */
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdint.h>
        #include <stdlib.h>
    #endif /* !defined( __cplusplus ) */
#endif /* !defined( SIXTRL_NO_SYSTEM_INCLUDES ) */

#if !defined( SIXTRL_NO_INCLUDES )
    #include "sixtracklib/common/definitions.h"
    #include "sixtracklib/common/track/definitions.h"

    #if defined( __cplusplus )
        #include "sixtracklib/common/buffer.hpp"
        #include "sixtracklib/common/particles/particles_addr.hpp"
    #endif /* defined( __cplusplus ) */

    #include "sixtracklib/common/buffer.h"
    #include "sixtracklib/common/buffer/assign_address_item.hpp"
    #include "sixtracklib/common/particles.h"
    #include "sixtracklib/common/particles/particles_addr.h"
    #include "sixtracklib/common/output/output_buffer.h"
    #include "sixtracklib/common/output/elem_by_elem_config.h"
    #include "sixtracklib/common/track/track_job_buffer_store.h"
#endif /* !defined( SIXTRL_NO_INCLUDES ) */

#if defined( __cplusplus )
namespace SIXTRL_CXX_NAMESPACE
{
    class TrackJobBase
    {
        public:

        using buffer_t               = SIXTRL_CXX_NAMESPACE::Buffer;
        using c_buffer_t             = ::NS(Buffer);
        using elem_by_elem_config_t  = ::NS(ElemByElemConfig);
        using elem_by_elem_order_t   = ::NS(elem_by_elem_order_t);
        using particle_index_t       = ::NS(particle_index_t);
        using size_type              = SIXTRL_CXX_NAMESPACE::track_job_size_t;
        using type_t                 = SIXTRL_CXX_NAMESPACE::track_job_type_t;
        using track_status_t         = SIXTRL_CXX_NAMESPACE::track_status_t;
        using status_t               = SIXTRL_CXX_NAMESPACE::arch_status_t;
        using assign_item_t          = SIXTRL_CXX_NAMESPACE::AssignAddressItem;
        using assign_item_key_t      = ::NS(TrackJobDestSrcBufferIds);
        using output_buffer_flag_t   = ::NS(output_buffer_flag_t);
        using object_type_id_t       = ::NS(object_type_id_t);
        using particles_addr_t       = ::NS(ParticlesAddr);

        using collect_flag_t = SIXTRL_CXX_NAMESPACE::track_job_collect_flag_t;
        using push_flag_t    = SIXTRL_CXX_NAMESPACE::track_job_push_flag_t;

        /* ----------------------------------------------------------------- */

        static constexpr size_type ILLEGAL_BUFFER_ID =
                    SIXTRL_CXX_NAMESPACE::ARCH_ILLEGAL_BUFFER_ID;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN static bool IsCollectFlagSet(
            collect_flag_t const haystack,
            collect_flag_t const needle ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN static size_type
        DefaultNumParticleSetIndices() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN static size_type const*
        DefaultParticleSetIndicesBegin() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN static size_type const*
        DefaultParticleSetIndicesEnd() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void clear();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void collect();
        SIXTRL_HOST_FN void collect( collect_flag_t const flags );

        SIXTRL_HOST_FN void collectParticles();
        SIXTRL_HOST_FN void collectBeamElements();
        SIXTRL_HOST_FN void collectOutput();
        SIXTRL_HOST_FN status_t collectParticlesAddresses();

        SIXTRL_HOST_FN void enableCollectParticles()  SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void disableCollectParticles() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool isCollectingParticles() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void enableCollectBeamElements()  SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void disableCollectBeamElements() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool isCollectingBeamElements() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void enableCollectOutput()  SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void disableCollectOutput() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool isCollectingOutput() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN collect_flag_t collectFlags() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN void setCollectFlags(
            collect_flag_t const flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool requiresCollecting() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void push( push_flag_t const flags );
        SIXTRL_HOST_FN void pushParticles();
        SIXTRL_HOST_FN void pushBeamElements();
        SIXTRL_HOST_FN void pushOutput();
        SIXTRL_HOST_FN void pushParticlesAddresses();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool canFetchParticleAddresses() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasParticleAddresses() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t fetchParticleAddresses();

        SIXTRL_HOST_FN status_t clearParticleAddresses(
            size_type const index = size_type{ 0 } );

        SIXTRL_HOST_FN status_t clearAllParticleAddresses();

        SIXTRL_HOST_FN particles_addr_t const* particleAddresses(
            size_type const index = size_type{ 0 } ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const*
        ptrParticleAddressesBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptrCParticleAddressesBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t*
            ptrParticleAddressesBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t*
            ptrCParticleAddressesBuffer() SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN track_status_t track(
            size_type const until_turn );

        SIXTRL_HOST_FN track_status_t trackElemByElem(
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN track_status_t trackLine(
            size_type const beam_elements_begin_index,
            size_type const beam_elements_end_index,
            bool const finish_turn = false );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual ~TrackJobBase() = default;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool reset(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer   = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            size_type const particle_set_index,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        template< typename ParSetIndexIter  >
        SIXTRL_HOST_FN bool reset(
            buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
            ParSetIndexIter  particle_set_indices_begin,
            ParSetIndexIter  particle_set_indices_end,
            buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
            buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const particle_set_index,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            size_type const num_particle_sets,
            size_type const* SIXTRL_RESTRICT particle_set_indices_begin,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        template< typename ParSetIndexIter  >
        SIXTRL_HOST_FN bool reset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            ParSetIndexIter  particle_set_indices_begin,
            ParSetIndexIter  particle_set_indices_end,
            c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer = nullptr,
            size_type const until_turn_elem_by_elem = size_type{ 0 } );

        SIXTRL_HOST_FN bool selectParticleSet(
            size_type const particle_set_index );

        SIXTRL_HOST_FN bool assignOutputBuffer(
            buffer_t& SIXTRL_RESTRICT_REF output_buffer );

        SIXTRL_HOST_FN bool assignOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN track_job_type_t type()          const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& typeStr()     const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrTypeStr()         const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasDeviceIdStr()            const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& deviceIdStr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrDeviceIdStr()     const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool hasConfigStr()              const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN std::string const& configStr()   const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN char const* ptrConfigStr()       const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type numParticleSets() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        particleSetIndicesBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        particleSetIndicesEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type particleSetIndex( size_type const n ) const;

        SIXTRL_HOST_FN size_type const*
        numParticlesInSetsBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        numParticlesInSetsEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type numParticlesInSet( size_type const n ) const;
        SIXTRL_HOST_FN size_type totalNumParticlesInSets() const;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN particle_index_t minParticleId() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN particle_index_t maxParticleId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t minElementId()  const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN particle_index_t maxElementId()  const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t
        minInitialTurnId() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t
        maxInitialTurnId() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN buffer_t* ptrParticlesBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const*
        ptrParticlesBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* ptrCParticlesBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptrCParticlesBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t* ptrBeamElementsBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const*
        ptrBeamElementsBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* ptrCBeamElementsBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptrCBeamElementsBuffer() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool can_fetch_particles_addr() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool has_particles_addr() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t fetch_particles_addr();
        SIXTRL_HOST_FN status_t clear_all_particles_addr();
        SIXTRL_HOST_FN status_t clear_particles_addr(
            size_type const index = size_type{ 0 } );

        SIXTRL_HOST_FN particles_addr_t const* particles_addr(
            size_type const index = size_type{ 0 } ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const*
        ptr_particles_addr_buffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptr_particles_addr_cbuffer() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN assign_item_t* add_assign_address_item(
            assign_item_t const& SIXTRL_RESTRICT_REF assign_item_to_add );

        SIXTRL_HOST_FN assign_item_t* add_assign_address_item(
            object_type_id_t const dest_type_id,
            size_type const dest_buffer_id,
            size_type const dest_elem_index,
            size_type const dest_pointer_offset,
            object_type_id_t const src_type_id,
            size_type const src_buffer_id,
            size_type const src_elem_index,
            size_type const src_pointer_offset );

        SIXTRL_HOST_FN status_t remove_assign_address_item(
            assign_item_key_t const& SIXTRL_RESTRICT_REF key,
            size_type const index_of_item_to_remove );

        SIXTRL_HOST_FN status_t remove_assign_address_item(
            assign_item_t const& SIXTRL_RESTRICT_REF item_to_remove );

        SIXTRL_HOST_FN bool has_assign_address_item( assign_item_t const&
            SIXTRL_RESTRICT_REF assign_item ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool has_assign_address_item(
            object_type_id_t const dest_type_id,
            size_type const dest_buffer_id,
            size_type const dest_elem_index,
            size_type const dest_pointer_offset,
            object_type_id_t const src_type_id,
            size_type const src_buffer_id,
            size_type const src_elem_index,
            size_type const src_pointer_offset ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type index_of_assign_address_item(
            object_type_id_t const dest_type_id,
            size_type const dest_buffer_id,
            size_type const dest_elem_index,
            size_type const dest_pointer_offset,
            object_type_id_t const src_type_id,
            size_type const src_buffer_id,
            size_type const src_elem_index,
            size_type const src_pointer_offset ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type index_of_assign_address_item(
            assign_item_t const& SIXTRL_RESTRICT_REF
                assign_item ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool has_assign_items(
            size_type const dest_buffer_id,
            size_type const src_buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type num_assign_items(
            size_type const dest_buffer_id,
            size_type const src_buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type total_num_assign_items() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN assign_item_t const* ptr_assign_address_item(
            assign_item_t const& SIXTRL_RESTRICT_REF
                assign_address_item ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN assign_item_t const* ptr_assign_address_item(
            size_type const dest_buffer_id,  size_type const src_buffer_id,
            size_type const assign_item_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN assign_item_t const* ptr_assign_address_item(
            object_type_id_t const dest_type_id,
            size_type const dest_buffer_id,
            size_type const dest_elem_index,
            size_type const dest_pointer_offset,
            object_type_id_t const src_type_id,
            size_type const src_buffer_id,
            size_type const src_elem_index,
            size_type const src_pointer_offset ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
            num_distinct_available_assign_address_items_dest_src_pairs() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type available_assign_address_items_dest_src_pairs(
            size_type const max_num_pairs,
            assign_item_key_t* pairs_begin ) const SIXTRL_NOEXCEPT;

        template< typename PairIter >
        SIXTRL_HOST_FN size_type available_assign_address_items_dest_src_pairs(
            PairIter pairs_begin, PairIter pairs_end ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* buffer_by_buffer_id(
            size_type const buffer_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const* buffer_by_buffer_id(
            size_type const buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool is_buffer_by_buffer_id(
            size_type const buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool is_raw_memory_by_buffer_id(
            size_type const buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*
        assign_items_begin( size_type const dest_buffer_id,
            size_type const src_buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN SIXTRL_BUFFER_OBJ_ARGPTR_DEC NS(Object) const*
        assign_items_end( size_type const dest_buffer_id,
            size_type const src_buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN assign_item_key_t const*
        assign_item_dest_src_begin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN assign_item_key_t const*
        assign_item_dest_src_end() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t commit_address_assignments();

        SIXTRL_HOST_FN status_t assign_all_addresses();
        SIXTRL_HOST_FN status_t assign_addresses(
            size_type const dest_buffer_id, size_type const src_buffer_id );

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN size_type
        stored_buffers_capacity() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t reserve_stored_buffers_capacity(
            size_type const capacity );

        SIXTRL_HOST_FN bool has_stored_buffers() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
            num_stored_buffers() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
            min_stored_buffer_id() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
            max_stored_buffer_id() const SIXTRL_NOEXCEPT;

        template< typename... Args >
        SIXTRL_HOST_FN size_type add_stored_buffer( Args&&... args );

        SIXTRL_HOST_FN bool owns_stored_buffer(
            size_type const buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t remove_stored_buffer(
            size_type const buffer_index );

        SIXTRL_HOST_FN buffer_t& stored_cxx_buffer(
            size_type const buffer_id );

        SIXTRL_HOST_FN buffer_t const& stored_cxx_buffer(
            size_type const buffer_id ) const;

        SIXTRL_HOST_FN buffer_t* ptr_stored_cxx_buffer(
            size_type const buffer_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const* ptr_stored_cxx_buffer(
            size_type const buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* ptr_stored_buffer(
            size_type const buffer_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const* ptr_stored_buffer(
            size_type const buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN status_t push_stored_buffer(
            size_type const buffer_id );

        SIXTRL_HOST_FN status_t collect_stored_buffer(
            size_type const buffer_id );

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasOutputBuffer()      const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool ownsOutputBuffer()     const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool hasElemByElemOutput()  const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool hasBeamMonitorOutput() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
        beamMonitorsOutputBufferOffset() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type
        elemByElemOutputBufferOffset() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN particle_index_t
        untilTurnElemByElem() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type numElemByElemTurns() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t* ptrOutputBuffer() SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN buffer_t const* ptrOutputBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* ptrCOutputBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
            ptrCOutputBuffer() const SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasBeamMonitors()      const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN size_type numBeamMonitors() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        beamMonitorIndicesBegin() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type const*
        beamMonitorIndicesEnd() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type beamMonitorIndex( size_type const n ) const;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN bool hasElemByElemConfig() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t const*
        ptrElemByElemConfigCBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t*
        ptrElemByElemConfigCBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t const*
        ptrElemByElemConfigBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t* ptrElemByElemConfigBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_config_t const*
        ptrElemByElemConfig() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_config_t*
        ptrElemByElemConfig() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN bool elemByElemRolling() const SIXTRL_NOEXCEPT;
        SIXTRL_HOST_FN bool defaultElemByElemRolling() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void setDefaultElemByElemRolling(
            bool is_rolling ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_order_t
        elemByElemOrder() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN elem_by_elem_order_t
        defaultElemByElemOrder() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void setDefaultElemByElemOrder(
            elem_by_elem_order_t const order ) SIXTRL_NOEXCEPT;

        /* ---------------------------------------------------------------- */

        SIXTRL_HOST_FN bool debugMode() const SIXTRL_NOEXCEPT;

        protected:

        using ptr_buffer_t   = std::unique_ptr< buffer_t >;
        using buffer_store_t = TrackJobBufferStore;

        /* ----------------------------------------------------------------- */

        SIXTRL_STATIC SIXTRL_HOST_FN void COPY_PTR_BUFFER(
            ptr_buffer_t& SIXTRL_RESTRICT_REF dest_ptr_buffer,
            ptr_buffer_t const& SIXTRL_RESTRICT_REF src_ptr_buffer );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN static collect_flag_t UnsetCollectFlag(
            collect_flag_t const haystack,
            collect_flag_t const needle ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN TrackJobBase(
            const char *const SIXTRL_RESTRICT type_str,
            track_job_type_t const type_id );

        SIXTRL_HOST_FN TrackJobBase( TrackJobBase const& other );
        SIXTRL_HOST_FN TrackJobBase( TrackJobBase&& other ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN TrackJobBase& operator=(
            TrackJobBase const& rhs );

        SIXTRL_HOST_FN TrackJobBase& operator=(
            TrackJobBase&& rhs ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN virtual void doClear();

        SIXTRL_HOST_FN virtual void doCollect( collect_flag_t const flags );
        SIXTRL_HOST_FN virtual void doPush( push_flag_t const flags );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual bool doPrepareParticlesStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_particles_buffer );

        SIXTRL_HOST_FN virtual bool doPrepareBeamElementsStructures(
            c_buffer_t* SIXTRL_RESTRICT ptr_beam_elem_buffer );

        SIXTRL_HOST_FN virtual bool doPrepareOutputStructures(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN virtual bool doAssignOutputBufferToBeamMonitors(
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            particle_index_t const min_turn_id,
            size_type const output_buffer_offset_index );

        SIXTRL_HOST_FN virtual bool doAssignOutputBufferToElemByElemConfig(
            elem_by_elem_config_t* SIXTRL_RESTRICT elem_by_elem_config,
            c_buffer_t* SIXTRL_RESTRICT output_buffer,
            size_type const output_buffer_offset_index );

        SIXTRL_HOST_FN virtual bool doReset(
            c_buffer_t* SIXTRL_RESTRICT particles_buffer,
            c_buffer_t* SIXTRL_RESTRICT beam_elem_buffer,
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN virtual bool doAssignNewOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer );

        SIXTRL_HOST_FN virtual size_type doAddStoredBuffer(
            buffer_store_t&& assigned_buffer_handle );

        SIXTRL_HOST_FN virtual status_t doRemoveStoredBuffer(
            size_type const buffer_id );

        SIXTRL_HOST_FN virtual status_t doPushStoredBuffer(
            size_type const buffer_id );

        SIXTRL_HOST_FN virtual status_t doCollectStoredBuffer(
            size_type const buffer_id );

        SIXTRL_HOST_FN virtual status_t doAddAssignAddressItem(
            assign_item_t const& SIXTRL_RESTRICT_REF assign_item,
            size_type* SIXTRL_RESTRICT ptr_item_index );

        SIXTRL_HOST_FN virtual status_t doRemoveAssignAddressItem(
            assign_item_key_t const& SIXTRL_RESTRICT_REF assign_item_key,
            size_type const index_of_item_to_remove );

        SIXTRL_HOST_FN virtual status_t doPerformAddressAssignments(
            assign_item_key_t const& SIXTRL_RESTRICT_REF assign_item_key );

        SIXTRL_HOST_FN virtual status_t doCommitAddressAssignments();

        SIXTRL_HOST_FN virtual status_t doRebuildAssignItemsBufferArg();

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual status_t doFetchParticleAddresses();

        SIXTRL_HOST_FN virtual status_t doClearParticleAddresses(
            size_type const index );

        SIXTRL_HOST_FN virtual status_t doClearAllParticleAddresses();

        SIXTRL_HOST_FN virtual track_status_t doTrackUntilTurn(
            size_type const until_turn );

        SIXTRL_HOST_FN virtual track_status_t doTrackElemByElem(
            size_type const until_turn_elem_by_elem );

        SIXTRL_HOST_FN virtual track_status_t doTrackLine(
            size_type const beam_elements_begin_index,
            size_type const beam_elements_end_index, bool const finish_turn );

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN virtual void doParseConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN void doSetDeviceIdStr(
            const char *const SIXTRL_RESTRICT device_id_str );

        SIXTRL_HOST_FN void doSetConfigStr(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN void doSetPtrParticleBuffer(
            buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrBeamElementsBuffer(
            buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrOutputBuffer(
            buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrCParticleBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrCBeamElementsBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetPtrCOutputBuffer(
            c_buffer_t* SIXTRL_RESTRICT ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetBeamMonitorOutputBufferOffset(
            size_type const output_buffer_offset ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetElemByElemOutputIndexOffset(
            size_type const target_num_output_turns ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetUntilTurnElemByElem(
            particle_index_t const until_turn_elem_by_elem ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetRequiresCollectFlag(
            bool const requires_collect_flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetBeamMonitorOutputEnabledFlag(
            bool const beam_monitor_flag ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetElemByElemOutputEnabledFlag(
            bool const elem_by_elem_flag ) SIXTRL_NOEXCEPT;

        template< typename Iter >
        SIXTRL_HOST_FN void doSetParticleSetIndices( Iter begin, Iter end,
            const c_buffer_t *const SIXTRL_RESTRICT pbuffer = nullptr );

        SIXTRL_HOST_FN void doInitDefaultParticleSetIndices();

        template< typename BeMonitorIndexIter >
        SIXTRL_HOST_FN void doSetBeamMonitorIndices(
            BeMonitorIndexIter begin, BeMonitorIndexIter end );

        SIXTRL_HOST_FN void doInitDefaultBeamMonitorIndices();

        SIXTRL_HOST_FN void doSetMinParticleId(
            particle_index_t const min_particle_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMaxParticleId(
            particle_index_t const max_particle_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMinElementId(
            particle_index_t const min_element_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMaxElementId(
            particle_index_t const max_element_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMinInitialTurnId(
            particle_index_t const min_initial_turn_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetMaxInitialTurnId(
            particle_index_t const max_initial_turn_id ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN buffer_t const*
        doGetPtrParticlesAddrBuffer() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_t* doGetPtrParticlesAddrBuffer() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredParticlesAddrBuffer(
            ptr_buffer_t&& ptr_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doSetHasParticleAddressesFlag(
            bool const has_particle_addresses ) SIXTRL_NOEXCEPT;

        /* ----------------------------------------------------------------- */

        SIXTRL_HOST_FN void doUpdateStoredOutputBuffer(
            ptr_buffer_t&& ptr_output_buffer ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doUpdateStoredElemByElemConfig(
            ptr_buffer_t&& ptr_config ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_store_t* doGetPtrBufferStore(
            size_type const buffer_id ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN buffer_store_t const* doGetPtrBufferStore(
            size_type const buffer_id ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type doGetStoredBufferSize() const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN size_type doFindAssingAddressItem( assign_item_t const&
            SIXTRL_RESTRICT_REF item_to_search ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN assign_item_t const* doGetAssignAddressItem(
            assign_item_key_t const& SIXTRL_RESTRICT_REF key,
            size_type const item_index ) const SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN assign_item_t* doGetAssignAddressItem(
            assign_item_key_t const& SIXTRL_RESTRICT_REF key,
            size_type const item_index ) SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN c_buffer_t* doGetPtrAssignAddressItemsBuffer(
            assign_item_key_t const& SIXTRL_RESTRICT_REF key ) SIXTRL_NOEXCEPT;


        private:

        using assing_item_map_t = std::map< assign_item_key_t, buffer_t,
            SIXTRL_CXX_NAMESPACE::TrackJobDestSrcBufferIdsLessCmp >;
        using assign_item_keys_list_t = std::vector< assign_item_key_t >;

        SIXTRL_HOST_FN void doClearBaseImpl() SIXTRL_NOEXCEPT;

        SIXTRL_HOST_FN void doParseConfigStrBaseImpl(
            const char *const SIXTRL_RESTRICT config_str );

        SIXTRL_HOST_FN status_t doClearParticleAddressesBaseImpl(
            size_type const index );

        SIXTRL_HOST_FN status_t doClearAllParticleAddressesBaseImpl();

        std::string                     m_type_str;
        std::string                     m_device_id_str;
        std::string                     m_config_str;

        assing_item_map_t               m_assign_address_items;
        std::vector< size_type >        m_particle_set_indices;
        std::vector< size_type >        m_num_particles_in_sets;
        std::vector< size_type >        m_beam_monitor_indices;
        std::vector< buffer_store_t >   m_stored_buffers;
        assign_item_keys_list_t         m_assign_item_keys;

        ptr_buffer_t                    m_my_output_buffer;
        ptr_buffer_t                    m_elem_by_elem_buffer;
        ptr_buffer_t                    m_particles_addr_buffer;

        buffer_t*   SIXTRL_RESTRICT     m_ptr_particles_buffer;
        buffer_t*   SIXTRL_RESTRICT     m_ptr_beam_elem_buffer;
        buffer_t*   SIXTRL_RESTRICT     m_ptr_output_buffer;

        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_particles_buffer;
        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_beam_elem_buffer;
        c_buffer_t* SIXTRL_RESTRICT     m_ptr_c_output_buffer;

        size_type                       m_be_mon_output_buffer_offset;
        size_type                       m_elem_by_elem_output_offset;
        size_type                       m_total_num_particles_in_sets;
        size_type                       m_num_stored_buffers;

        type_t                          m_type_id;
        elem_by_elem_order_t            m_default_elem_by_elem_order;

        particle_index_t                m_min_particle_id;
        particle_index_t                m_max_particle_id;

        particle_index_t                m_min_element_id;
        particle_index_t                m_max_element_id;

        particle_index_t                m_min_initial_turn_id;
        particle_index_t                m_max_initial_turn_id;
        particle_index_t                m_until_turn_elem_by_elem;

        collect_flag_t                  m_collect_flags;
        bool                            m_requires_collect;
        bool                            m_default_elem_by_elem_rolling;
        bool                            m_has_beam_monitor_output;
        bool                            m_has_elem_by_elem_output;
        bool                            m_has_particle_addresses;
        bool                            m_debug_mode;
    };
}

typedef SIXTRL_CXX_NAMESPACE::TrackJobBase NS(TrackJobBase);

#else /* defined( __cplusplus ) && !defined( _GPUCODE ) */

typedef void NS(TrackJobBase);

#endif /* defined( __cplusplus ) && !defined( _GPUCODE ) */

#if !defined( _GPUCODE )
#if defined( __cplusplus )

/* ------------------------------------------------------------------------- */
/* ---- implementation of inline and template C++ methods/functions     ---- */
/* ------------------------------------------------------------------------- */

namespace SIXTRL_CXX_NAMESPACE
{
    template< typename ParSetIndexIter  >
    SIXTRL_HOST_FN bool TrackJobBase::reset(
        TrackJobBase::buffer_t& SIXTRL_RESTRICT_REF particles_buffer,
        ParSetIndexIter  particle_set_indices_begin,
        ParSetIndexIter  particle_set_indices_end,
        TrackJobBase::buffer_t& SIXTRL_RESTRICT_REF beam_elements_buffer,
        buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        size_type const until_turn_elem_by_elem )
    {
        using c_buffer_t = TrackJobBase::c_buffer_t;

        bool success = false;

        this->doClear();

        c_buffer_t* ptr_pb  = particles_buffer.getCApiPtr();
        c_buffer_t* ptr_eb  = beam_elements_buffer.getCApiPtr();
        c_buffer_t* ptr_out = ( ptr_output_buffer != nullptr )
            ? ptr_output_buffer->getCApiPtr() : nullptr;

        if( ( particle_set_indices_begin !=
              particle_set_indices_end ) &&
            ( std::distance( particle_set_indices_begin,
                particle_set_indices_end ) > std::ptrdiff_t{ 0 } ) )
        {
            this->doSetParticleSetIndices(
                particle_set_indices_begin, particle_set_indices_end, ptr_pb );
        }
        else
        {
            this->doInitDefaultBeamMonitorIndices();
        }

        success = this->doReset(
            ptr_pb, ptr_eb, ptr_out, until_turn_elem_by_elem );

        if( success )
        {
            this->doSetPtrParticleBuffer( &particles_buffer );
            this->doSetPtrBeamElementsBuffer( &beam_elements_buffer );

            if( ( ptr_out != nullptr ) && ( this->hasOutputBuffer() ) )
            {
                this->doSetPtrOutputBuffer( ptr_output_buffer );
            }
        }

        return success;
    }

    template< typename ParSetIndexIter  >
    SIXTRL_HOST_FN bool TrackJobBase::reset(
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT particles_buffer,
        ParSetIndexIter  particle_set_indices_begin,
        ParSetIndexIter  particle_set_indices_end,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT beam_elements_buffer,
        TrackJobBase::c_buffer_t* SIXTRL_RESTRICT ptr_output_buffer,
        TrackJobBase::size_type const until_turn_elem_by_elem )
    {
        this->doClear();

        if( ( particle_set_indices_begin !=
              particle_set_indices_end ) &&
            ( std::distance( particle_set_indices_begin,
                particle_set_indices_end ) > std::ptrdiff_t{ 0 } ) )
        {
            this->doSetParticleSetIndices( particle_set_indices_begin,
                particle_set_indices_end, particles_buffer );
        }
        else
        {
            this->doInitDefaultBeamMonitorIndices();
        }

        bool success = this->doReset( particles_buffer, beam_elements_buffer,
            ptr_output_buffer, until_turn_elem_by_elem );

        /*
        if( success )
        {
            this->doSetPtrCParticleBuffer( particles_buffer );
            this->doSetPtrCBeamElementsBuffer( beam_elements_buffer );
        }
        */

        return success;
    }

    template< typename... Args >
    TrackJobBase::size_type TrackJobBase::add_stored_buffer( Args&&... args )
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using _this_t = st::TrackJobBase;

        _this_t::buffer_store_t temp_buffer_store(
            std::forward< Args >( args )... );

        _this_t::size_type const buffer_id =
            this->doAddStoredBuffer( std::move( temp_buffer_store ) );

        return buffer_id;
    }

    template< typename PairIter >
    SIXTRL_HOST_FN TrackJobBase::size_type
    TrackJobBase::available_assign_address_items_dest_src_pairs(
        PairIter out_it, PairIter out_end ) const SIXTRL_NOEXCEPT
    {
        namespace st = SIXTRL_CXX_NAMESPACE;
        using _this_t = st::TrackJobBase;
        using size_t = _this_t::size_type;
        size_t num_pairs = size_t{ 0 };

        if( !this->m_assign_address_items.empty() )
        {
            SIXTRL_ASSERT( this->m_assign_address_items.size() ==
                           this->m_assign_item_keys.size() );

            auto it = this->m_assign_item_keys.begin();
            auto end = this->m_assign_item_keys.end();

            while( ( out_it != out_end ) && ( it != end ) )
            {
                *out_it = *it;
                ++num_pairs;
                ++out_it;
                ++it;
            }
        }

        std::fill( out_it, out_end, _this_t::assign_item_key_t{
            _this_t::ILLEGAL_BUFFER_ID, _this_t::ILLEGAL_BUFFER_ID } );

        return num_pairs;
    }

    template< typename ParSetIndexIter >
    SIXTRL_HOST_FN void TrackJobBase::doSetParticleSetIndices(
        ParSetIndexIter begin, ParSetIndexIter end,
        const TrackJobBase::c_buffer_t *const SIXTRL_RESTRICT pbuffer )
    {
        using diff_t   = std::ptrdiff_t;
        using size_t   = TrackJobBase::size_type;
        using obj_it_t = SIXTRL_BUFFER_DATAPTR_DEC ::NS(Object) const*;
        using nparticles_t = ::NS(particle_num_elements_t);

        diff_t const temp_len = std::distance( begin, end );

        if( temp_len >= diff_t{ 0 } )
        {
            size_t total_num_particles_in_sets = size_t{ 0 };
            this->m_particle_set_indices.clear();
            this->m_num_particles_in_sets.clear();

            if( temp_len > diff_t{ 0 } )
            {
                this->m_particle_set_indices.clear();
                this->m_particle_set_indices.reserve(
                    static_cast< size_t >( temp_len ) );

                if( pbuffer == nullptr )
                {
                    this->m_particle_set_indices.assign( begin, end );
                }
                else
                {
                    size_t const SETS_IN_BUFFER =
                        ::NS(Buffer_get_num_of_objects)( pbuffer );

                    obj_it_t obj = nullptr;

                    for( ParSetIndexIter it = begin ; it != end ; ++it )
                    {
                        obj = ::NS(Buffer_get_const_object)( pbuffer, *it );

                        if( ( *it < SETS_IN_BUFFER ) && ( obj != nullptr ) &&
                            ( ::NS(Object_get_type_id)( obj ) ==
                                ::NS(OBJECT_TYPE_PARTICLE) ) )
                        {
                            this->m_particle_set_indices.push_back( *it );
                        }
                    }
                }

                std::sort( this->m_particle_set_indices.begin(),
                           this->m_particle_set_indices.end() );

                this->m_particle_set_indices.erase( std::unique(
                    this->m_particle_set_indices.begin(),
                    this->m_particle_set_indices.end() ),
                    this->m_particle_set_indices.end() );

                if( !this->m_particle_set_indices.empty() )
                {
                    for( auto const pset_idx : this->m_particle_set_indices )
                    {
                        auto pset = ::NS(Particles_buffer_get_const_particles)(
                            pbuffer, pset_idx );

                        SIXTRL_ASSERT( pset != nullptr );

                        nparticles_t const nparticles =
                            ::NS(Particles_get_num_of_particles)( pset );

                        SIXTRL_ASSERT( nparticles >= nparticles_t{ 0 } );

                        this->m_num_particles_in_sets.push_back( nparticles );
                        total_num_particles_in_sets += nparticles;
                    }

                    SIXTRL_ASSERT( this->m_num_particles_in_sets.size() ==
                                   this->m_particle_set_indices.size() );

                    this->m_total_num_particles_in_sets =
                        total_num_particles_in_sets;
                }
            }
        }

        return;
    }

    template< typename BeMonitorIndexIter >
    SIXTRL_HOST_FN void TrackJobBase::doSetBeamMonitorIndices(
        BeMonitorIndexIter begin, BeMonitorIndexIter end )
    {
        using diff_t = std::ptrdiff_t;
        using size_t = TrackJobBase::size_type;

        diff_t const temp_len = std::distance( begin, end );

        if( temp_len >= diff_t{ 0 } )
        {
            this->m_beam_monitor_indices.clear();

            if( temp_len > diff_t{ 0 } )
            {
                this->m_beam_monitor_indices.reserve(
                    static_cast< size_t >( temp_len ) );

                this->m_beam_monitor_indices.assign( begin, end );

                std::sort( this->m_beam_monitor_indices.begin(),
                           this->m_beam_monitor_indices.end() );

                this->m_beam_monitor_indices.erase( std::unique(
                    this->m_beam_monitor_indices.begin(),
                    this->m_beam_monitor_indices.end() ),
                    this->m_beam_monitor_indices.end() );
            }
        }

        return;
    }

    /* --------------------------------------------------------------------- */

    SIXTRL_INLINE bool TrackJobBase::IsCollectFlagSet(
        TrackJobBase::collect_flag_t const flag_set,
        TrackJobBase::collect_flag_t const flag ) SIXTRL_NOEXCEPT
    {
        return ( ( flag_set & flag ) == flag );
    }

    SIXTRL_INLINE TrackJobBase::collect_flag_t
    TrackJobBase::UnsetCollectFlag(
        TrackJobBase::collect_flag_t const flag_set,
        TrackJobBase::collect_flag_t const flag ) SIXTRL_NOEXCEPT
    {
        return flag_set & ~flag;
    }
}

#endif /* defined( __cplusplus ) */


#endif /* !defined( _GPUCODE ) */

#endif /* SIXTRACKLIB_SIXTRACKLIB_COMMON_INTERNAL_TRACK_JOB_BASE_H__ */
/*end: sixtracklib/common/internal/track_job_base.h */
