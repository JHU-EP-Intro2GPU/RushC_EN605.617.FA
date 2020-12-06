#ifndef UTILS_UNITS_HPP
#define UTILS_UNITS_HPP

// C
#include <stddef.h>

// C++
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

namespace utils
{
    //--------------------------------------------------------------------------
    // Memory Sizes
    //--------------------------------------------------------------------------

    /**
     * Represents a size of memory in terms of some unit.
     */
    class MemorySize
    {
    public:
        /**
         * Construct a memory size.
         * 
         * @param[in] count          The number of units.
         * @param[in] bytes_per_unit The number of bytes each unit contains.
         * @param[in] suffix         The suffix for the unit.
         */
        explicit constexpr MemorySize(size_t count, size_t bytes_per_unit, const char* suffix) : 
            m_count(count),
            m_bytes_per_unit(bytes_per_unit),
            m_suffix(suffix)
        {
            // Empty
        }

        /// The number of bytes represented by this size.
        constexpr size_t bytes() const
        {
            return this->m_count * this->m_bytes_per_unit;
        }

        /// The number of units.
        constexpr size_t count() const
        {
            return this->m_count;
        }

        /// The suffix for the unit.
        constexpr const char* suffix() const
        {
            return this->m_suffix;
        }

    private:
        size_t m_count;
        size_t m_bytes_per_unit;
        const char* m_suffix;
    };

    /// Custom literals for common units.
    inline namespace literals
    {
        constexpr MemorySize operator ""_B(unsigned long long count) { return MemorySize(count, 1, "B"); }
        constexpr MemorySize operator ""_kB(unsigned long long count) { return MemorySize(count, 1024, "kB"); }
        constexpr MemorySize operator ""_MB(unsigned long long count) { return MemorySize(count, 1024*1024, "MB"); }
        constexpr MemorySize operator ""_GB(unsigned long long count) { return MemorySize(count, 1024*1024*1024, "GB"); }
    }

    /// Output format for memory sizes.
    inline namespace operators
    {
        std::ostream& operator<<(std::ostream& out, const MemorySize& memory)
        {
            std::stringstream text;
            text << memory.count() << " " << memory.suffix();
            return out << text.str();
        }
    }

    //--------------------------------------------------------------------------
    // Time
    //--------------------------------------------------------------------------

    template<typename T>
    constexpr const char* units(T);

    /// Units for common std::chrono::duration specializations
    template<> constexpr const char* units(std::chrono::hours) { return "h"; };
    template<> constexpr const char* units(std::chrono::minutes) { return "m"; };
    template<> constexpr const char* units(std::chrono::seconds) { return "s"; };
    template<> constexpr const char* units(std::chrono::milliseconds) { return "ms"; };
    template<> constexpr const char* units(std::chrono::microseconds) { return "us"; };
    template<> constexpr const char* units(std::chrono::nanoseconds) { return "ns"; };

    /// Output format for common std::chrono::duration specializations
    inline namespace operators
    {
        template<typename Rep, typename Period>
        std::ostream& operator<<(std::ostream& out, const std::chrono::duration<Rep, Period>& time)
        {
            std::stringstream text;
            text << time.count() << utils::units(time);
            return out << text.str();
        }
    }

} // namespace utils

#endif // UTILS_UNITS_HPP