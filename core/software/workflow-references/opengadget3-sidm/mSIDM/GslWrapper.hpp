/**
 * @file
 *
 * This file is part of the N-body/SPH code OpenGadget3.
 *
 * @brief A GSL Function Wrapper that allows functions to be called with arguments from a json
 *
 * @author OpenGadget3-development team, in particular Marc Wiertel and Moritz S. Fischer.
 *
 * @copyright Copyright (c) 2026, OpenGadget3-development team lead by Klaus Dolag and Stefano Borgani
 * (see change logs). All rights reserved.
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License v3 or later.
 * See <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <tuple>
#include <string>
#include <array>
#include <iostream>
#define JSON_ASSERT(x) // suppress redefinition of assert
#include <nlohmann/json.hpp>
#include <gsl/gsl_integration.h>
#include <type_traits>
#include <cxxabi.h>
#include <stdlib.h>

// TODO@MW: If this class is to be used outside of SIDM, need to change namespace or move to global namespace
namespace scatter
{

using json = nlohmann::json;

struct GSLWrapperBase
{
    virtual ~GSLWrapperBase() = default;
    virtual void set_velocity(double value) = 0;
};

struct FunctionEntry
{
    std::function<gsl_function(const json &)> make;
};

inline std::unordered_map<std::string, FunctionEntry> GslFactoryRegistry;
inline std::unordered_map<std::string, std::shared_ptr<GSLWrapperBase>> GslWrapperRegistry;

template <auto Func> struct function_metadata;

template <typename T> struct function_traits;

// Stores information about function signature
template <typename R, typename... Args> struct function_traits<R (*)(Args...)>
{
    using return_type = R;
    using arg_types = std::tuple<Args...>;
    static constexpr std::size_t arg_count = sizeof...(Args);
};

// empty tuple --> first argument == void
template <typename Tuple> struct first_arg_type
{
    using type = void;
};

// Get first type of tuple
template <typename T, typename... Ts> struct first_arg_type<std::tuple<T, Ts...>>
{
    using type = T;
};

// Check signature of function
template <auto Func> constexpr bool check_function()
{
    using traits = function_traits<decltype(Func)>;
    constexpr bool a = std::is_same_v<typename traits::return_type, double>;
    constexpr bool b = traits::arg_count >= 1;
    constexpr bool c = std::is_same_v<typename first_arg_type<typename traits::arg_types>::type, double>;
    return a && b && c;
}

// Count commas, because c++17 cannot infer NTTP
template <std::size_t N> constexpr std::size_t count_commas(const char (&s)[N])
{
    std::size_t count = 0;
    for (std::size_t i = 0; i < N - 1; ++i)
    {
        if (s[i] == ',')
            ++count;
    }
    return count;
}

// split string of arguments (from macro) to array of arguments
template <std::size_t N, std::size_t param_count> constexpr auto split(const char (&s)[N])
{
    const auto str = std::string_view{s, N - 1}; // exclude zero byte

    struct SplitResult
    {
        std::array<std::string_view, param_count> arguments{};
        std::size_t count = param_count;
    };

    SplitResult result{};
    std::size_t start = 0;
    std::size_t idx = 0;

    for (std::size_t i = 0; i <= str.size(); ++i)
    {
        if (i == str.size() || str[i] == ',')
        {
            std::size_t ss = start, e = i;

            // trim spaces
            while (ss < e && (str[ss] == ' ' || str[ss] == '\t'))
                ++ss;
            while (e > ss && (str[e - 1] == ' ' || str[e - 1] == '\t'))
                --e;

            result.arguments[idx++] = str.substr(ss, e - ss);
            start = i + 1;
        }
    }

    return result;
}

#define REGISTER_GSL_FUNCTION(FUNC)                                                                                    \
    namespace                                                                                                          \
    {                                                                                                                  \
    struct Register_##FUNC                                                                                             \
    {                                                                                                                  \
        Register_##FUNC()                                                                                              \
        {                                                                                                              \
            register_function<FUNC>(#FUNC);                                                                            \
        }                                                                                                              \
    };                                                                                                                 \
    static Register_##FUNC instance_##FUNC;                                                                            \
    }

/**
 * Register a static function of signature double(double, ...). This will make this function available for the gsl
 * wrapper. Note: Has to be called within "scatter" namespace. The first argument is always the scattering angle. An
 * argument called 'vrel' will always be considered the velocity and changed automatically during the computation of the
 * table.
 * Usage: REGISTER_FUNC(function name, arg1 name, arg2 name, arg3 name, ...)
 */
#define REGISTER_FUNC(func, ...)                                                                                       \
    static constexpr auto func##_params = split<sizeof(#__VA_ARGS__), count_commas(#__VA_ARGS__) + 1>(#__VA_ARGS__);   \
    template <> struct function_metadata<func>                                                                         \
    {                                                                                                                  \
        static constexpr auto params = func##_params.arguments;                                                        \
    };                                                                                                                 \
    static_assert(check_function<&func>(), "Function does not have correct signature. Use double(double, ...)");       \
    REGISTER_GSL_FUNCTION(func)

template <typename Tuple, size_t... I> constexpr auto tuple_tail_impl(Tuple &&t, std::index_sequence<I...>)
{
    return std::make_tuple(std::get<I + 1>(std::forward<Tuple>(t))...);
}

// returns a new tuple without the first element.
template <typename Tuple> constexpr auto tuple_tail(Tuple &&t)
{
    constexpr size_t N = std::tuple_size<std::decay_t<Tuple>>::value;
    static_assert(N > 0, "Tuple must have at least one element to drop the head");
    return tuple_tail_impl(std::forward<Tuple>(t), std::make_index_sequence<N - 1>{});
}

template <typename Tuple, auto Func, size_t... Is>
void fill_tuple(Tuple &paramTuple, const json &j, std::index_sequence<Is...>)
{
    constexpr auto &params = function_metadata<Func>::params;
    ((std::get<Is>(paramTuple) = j.at(params[Is + 1]).template get<std::tuple_element_t<Is, Tuple>>()), ...);
}

// Convert input json into a tuple, using the ordering given from function signature
template <auto Func> auto toTuple(const json &j)
{
    using traits = function_traits<decltype(Func)>;
    using ArgTuple = typename traits::arg_types;

    // Reduce signature by first argument, because GSL expects the first argument separately
    ArgTuple jpars;
    auto paramTuple = tuple_tail(std::forward<ArgTuple>(jpars));
    using ResultTuple = decltype(paramTuple);

    constexpr size_t N = std::tuple_size_v<ResultTuple>; // size of const parameters

    fill_tuple<ResultTuple, Func>(paramTuple, j, std::make_index_sequence<N>{});
    return paramTuple;
}

template <auto Func> constexpr int velocityIndex()
{
    constexpr auto &params = function_metadata<Func>::params;
    int i = 0;
    for (const auto &x : params)
    {
        if (x == "vrel")
        {
            return i - 1;
        }
        ++i;
    }
    return -1;
}

// Check if json is compatible with function signature === all arguments have to be specified. Additional fields are
// ignored (including first argument)
template <auto Func> bool doesJsonFit(const json &j)
{
    constexpr auto &params = function_metadata<Func>::params;

    // Use params[i].name to check j keys etc.

    for (size_t i = 1; i < params.size(); ++i)
    {
        // Note: allows non-numerical values
        if (!j.contains(std::string(params[i]))) // || !j[params[i]].is_number())
        {
            std::cout << "GslWrapper.hpp: supplied json is missing field: '" << std::string(params[i]) << "'"
                      << std::endl;
            return false;
        }
    }
    return true;
}

// Print out tuples
template <typename T> std::string type_name()
{
    int status;
    char *demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
    std::string result = (status == 0) ? demangled : typeid(T).name();
    std::free(demangled);
    return result;
}

template <typename Tuple, std::size_t... Is> void print_tuple_impl(const Tuple &t, std::index_sequence<Is...>)
{
    ((std::cout << "Element " << Is << ": value = " << std::get<Is>(t)
                << ", type = " << type_name<decltype(std::get<Is>(t))>() << '\n'),
     ...);
}

template <typename... Args> void print_tuple(const std::tuple<Args...> &t)
{
    print_tuple_impl(t, std::index_sequence_for<Args...>{});
}

template <auto Func, typename Tuple> struct GSLWrapper : GSLWrapperBase
{
    using TupleType = Tuple;

    Tuple params;
    static constexpr int velocity_index = velocityIndex<Func>();

    // Needed because params cannot be default constructed
    GSLWrapper(Tuple p) : params(std::move(p))
    {
    }

    void set_velocity(double value) override
    {
        if constexpr (velocity_index >= 0)
        {
            // std::cout << "Set velocity to " << value << std::endl;
            std::get<velocity_index>(params) = value;
        }
    }

    Tuple &getParamsTyped()
    {
        return params;
    }

    static double call(double x, void *p)
    {
        auto &wrapper = *static_cast<GSLWrapper *>(p);
        auto full_args = std::tuple_cat(std::make_tuple(x), wrapper.params);
        return std::apply(Func, full_args);
    }

    gsl_function get()
    {
        gsl_function F;
        F.function = &GSLWrapper::call;
        F.params = this;
        return F;
    }
};

template <auto Func> auto make_gsl_wrapper(const json &j)
{
    bool ret = doesJsonFit<Func>(j);
    if (!ret)
    {
        // TODO@MW: Return optional instead?
        std::cerr << "GslWrapper.hpp: Invalid json supplied\n";
        std::abort();
    }
    auto args = toTuple<Func>(j);
    return GSLWrapper<Func, decltype(args)>{std::move(args)};
}

template <auto Func> auto make_gsl_wrapper_ptr(const json &j)
{
    bool ret = doesJsonFit<Func>(j);
    if (!ret)
    {
        std::cerr << "GslWrapper.hpp: Invalid json supplied\n";
        std::abort();
    }
    auto args = toTuple<Func>(j);
    auto wrapper = std::make_shared<GSLWrapper<Func, decltype(args)>>(std::move(args));
    return wrapper;
}

template <auto Func> void register_function(const std::string &name)
{
    GslFactoryRegistry[name] = {[name](const json &j) {
        auto wrapper_ptr = make_gsl_wrapper_ptr<Func>(j);
        GslWrapperRegistry[name] = wrapper_ptr;
        return wrapper_ptr->get();
    }};
}

} // namespace scatter
