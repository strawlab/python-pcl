# -*- coding: utf-8 -*-
cimport pcl_defs as cpp
from libcpp cimport bool

###############################################################################
# Types
###############################################################################

# cdef extern from "boost/signals2.hpp" namespace "boost" nogil:
# http://www.boost.org/doc/libs/1_63_0/doc/html/signals2/reference.html#header.boost.signals2.connection_hpp
cdef extern from "boost/signals2.hpp" namespace "boost::signals2" nogil:
    cdef cppclass connection[T]:
        connection()

# cdef extern from "boost/signals2.hpp" namespace "boost::signals2" nogil:
#     cdef void swap(connection&, connection&)

# cdef extern from "boost/signals2.hpp" namespace "boost::signals2" nogil:
#     cdef class scoped_connection;

# cdef extern from "boost/signals2.hpp" namespace "boost::signals2" nogil:
#     cdef class deconstruct_access;
#     cdef class postconstructor_invoker;
#     cdef template<typename T> postconstructor_invoker<T> deconstruct();
#     cdef template<typename T, typename A1> postconstructor_invoker<T> deconstruct(const A1 &);
#     cdef template<typename T, typename A1, typename A2> postconstructor_invoker<T> deconstruct(const A1 &, const A2 &);
#     cdef template<typename T, typename A1, typename A2, ..., typename AN> postconstructor_invoker<T> deconstruct(const A1 &, const A2 &, ..., const AN &);

# cdef extern from "boost/signals2.hpp" namespace "boost::signals2" nogil:
#     cdef class dummy_mutex;

# Header <boost/signals2/last_value.hpp>
#     cdef template<typename T> class last_value;
#     cdef template<> class last_value<void>;
#     cdef class no_slots_error;

# Header <boost/signals2/mutex.hpp>
#     cdef class mutex;

# Header <boost/signals2/optional_last_value.hpp>
#     cdef template<typename T> class optional_last_value;
#     cdef template<> class optional_last_value<void>;

# Header <boost/signals2/shared_connection_block.hpp>
#     cdef class shared_connection_block;

# Header <boost/signals2/signal.hpp>
    # cdef enum connect_position { at_front, at_back };

    # template<typename Signature, 
    #          typename Combiner = boost::signals2::optional_last_value<R>, 
    #          typename Group = int, typename GroupCompare = std::less<Group>, 
    #          typename SlotFunction = boost::function<Signature>, 
    #          typename ExtendedSlotFunction = boost::function<R (const connection &, T1, T2, ..., TN)>, 
    #          typename Mutex = boost::signals2::mutex> 
    #   class signal;
    
    # template<typename Signature, typename Combiner, typename Group, 
    #          typename GroupCompare, typename SlotFunction, 
    #          typename ExtendedSlotFunction, typename Mutex> 
    #   void swap(signal<Signature, Combiner, Group, GroupCompare, SlotFunction, ExtendedSlotFunction, Mutex>&, 
    #             signal<Signature, Combiner, Group, GroupCompare, SlotFunction, ExtendedSlotFunction, Mutex>&);

# Header <boost/signals2/signal_base.hpp>
    # cdef class signal_base;


# Header <boost/signals2/signal_type.hpp>
    # template<typename A0, typename A1 = boost::parameter::void_, 
    #          typename A2 = boost::parameter::void_, 
    #          typename A3 = boost::parameter::void_, 
    #          typename A4 = boost::parameter::void_, 
    #          typename A5 = boost::parameter::void_, 
    #          typename A6 = boost::parameter::void_> 
    #   class signal_type;
    
    # namespace keywords {
    #   template<typename Signature> class signature_type;
    #   template<typename Combiner> class combiner_type;
    #   template<typename Group> class group_type;
    #   template<typename GroupCompare> class group_compare_type;
    #   template<typename SlotFunction> class slot_function_type;
    #   template<typename ExtendedSlotFunction> class extended_slot_function_type;
    #   template<typename Mutex> class mutex_type;
    # 


# Header <boost/signals2/slot.hpp>
    # template<typename Signature, typename SlotFunction = boost::function<R (T1, T2, ..., TN)> > class slot;

# Header <boost/signals2/slot_base.hpp>
    # cdef class slot_base;
    # cdef class expired_slot;

# Header <boost/signals2/trackable.hpp>
    # cdef class trackable;


###############################################################################
# Enum
###############################################################################

###############################################################################
# Activation
###############################################################################
