#ifndef XTGEO_LOGGING_HPP_
#define XTGEO_LOGGING_HPP_
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <atomic>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace xtgeo::logging {

// Logger class for logging messages to Python's logging module
// This class is thread-safe and can be used in a multi-threaded environment, inlcuding
// OMP, but possibly be careful in parallel regions

// It uses Python's Global Interpreter Lock (GIL) to ensure thread safety
// and to avoid issues with Python's memory management
// The logger can be used to log messages at different levels: debug, info, warning,
// error, and critical
// The logger name is passed as a parameter to the constructor, and it is used to create
// a logger object in Python's logging module.

// Thread-local storage for Python state
class ThreadLocalPythonState
{
public:
    ThreadLocalPythonState() : m_initialized(false) {}

    // Initialize Python state for the current thread if needed
    void ensure_initialized()
    {
        if (!m_initialized) {
            // Initialize Python for this thread if needed
            if (!Py_IsInitialized()) {
                PyGILState_STATE gstate = PyGILState_Ensure();
                PyGILState_Release(gstate);
            }
            m_initialized = true;
        }
    }

private:
    bool m_initialized;
};

// Logger class with explicit OMP support
class Logger
{
public:
    explicit Logger(const std::string &name) : m_logger_name(name)
    {
        initialize_python_objects();
    }

    // Delete copy constructor and assignment operator
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;

    // Delete move operations as well since std::mutex is not movable
    Logger(Logger &&) = delete;
    Logger &operator=(Logger &&) = delete;

    // Initialize or reinitialize Python objects
    void initialize_python_objects()
    {
        py::gil_scoped_acquire acquire;
        py::object logging = py::module::import("logging");
        m_logger = logging.attr("getLogger")(m_logger_name);
        m_debug_level = logging.attr("DEBUG");
        m_info_level = logging.attr("INFO");
        m_warning_level = logging.attr("WARNING");
        m_error_level = logging.attr("ERROR");
        m_critical_level = logging.attr("CRITICAL");

        // Pre-populate the level map
        if (m_level_map.empty()) {
            m_level_map["debug"] = "debug";
            m_level_map["info"] = "info";
            m_level_map["warning"] = "warning";
            m_level_map["error"] = "error";
            m_level_map["critical"] = "critical";
        }
    }

    template<typename... Args>
    void log(const std::string &level, const std::string &format_string, Args... args)
    {
        std::string formatted_message;

        // Use fmt library for formatting
        try {
            formatted_message = fmt::format("[C++: {}] {}", m_logger_name,
                                            fmt::format(format_string, args...));
        } catch (const fmt::format_error &e) {
            // Fallback to traditional formatting if the format string is invalid
            std::ostringstream oss;
            ((oss << args << " "), ...);
            formatted_message =
              fmt::format("[C++: {}] {} {}", m_logger_name, format_string, oss.str());
        }

        // Thread-local initialization
        thread_local ThreadLocalPythonState tls;
        tls.ensure_initialized();

        // Acquire mutex for thread safety when accessing Python
        std::lock_guard<std::mutex> lock(m_log_mutex);

        // Call the appropriate logger method with GIL
        py::gil_scoped_acquire acquire;

        // Ensure Python objects are valid (they might not be in OMP contexts)
        if (!m_logger) {
            initialize_python_objects();
        }

        auto it = m_level_map.find(level);
        if (it != m_level_map.end()) {
            m_logger.attr(it->second.c_str())(formatted_message);
        } else {
            m_logger.attr("info")(formatted_message);
        }
    }

    template<typename... Args>
    void debug(const std::string &message, Args... args)
    {
        // Check if debug level is enabled before formatting message
        bool enabled = false;
        {
            py::gil_scoped_acquire gil;
            if (!m_logger) {
                initialize_python_objects();
            }
            enabled = m_logger.attr("isEnabledFor")(m_debug_level).cast<bool>();
        }

        if (!enabled) {
            return;
        }
        log("debug", message, args...);
    }

    // Other level methods similar to debug()...
    template<typename... Args>
    void info(const std::string &message, Args... args)
    {
        bool enabled = false;
        {
            py::gil_scoped_acquire gil;
            if (!m_logger) {
                initialize_python_objects();
            }
            enabled = m_logger.attr("isEnabledFor")(m_info_level).cast<bool>();
        }

        if (!enabled) {
            return;
        }
        log("info", message, args...);
    }

    template<typename... Args>
    void warning(const std::string &message, Args... args)
    {
        bool enabled = false;
        {
            py::gil_scoped_acquire gil;
            if (!m_logger) {
                initialize_python_objects();
            }
            enabled = m_logger.attr("isEnabledFor")(m_warning_level).cast<bool>();
        }

        if (!enabled) {
            return;
        }
        log("warning", message, args...);
    }

    template<typename... Args>
    void error(const std::string &message, Args... args)
    {
        bool enabled = false;
        {
            py::gil_scoped_acquire gil;
            if (!m_logger) {
                initialize_python_objects();
            }
            enabled = m_logger.attr("isEnabledFor")(m_error_level).cast<bool>();
        }

        if (!enabled) {
            return;
        }
        log("error", message, args...);
    }

    template<typename... Args>
    void critical(const std::string &message, Args... args)
    {
        bool enabled = false;
        {
            py::gil_scoped_acquire gil;
            if (!m_logger) {
                initialize_python_objects();
            }
            enabled = m_logger.attr("isEnabledFor")(m_critical_level).cast<bool>();
        }

        if (!enabled) {
            return;
        }
        log("critical", message, args...);
    }

private:
    std::string m_logger_name;
    py::object m_logger;
    py::object m_debug_level;
    py::object m_info_level;
    py::object m_warning_level;
    py::object m_error_level;
    py::object m_critical_level;
    std::mutex m_log_mutex;
    std::unordered_map<std::string, std::string> m_level_map;
};

// OMP-safe Singleton manager for loggers
class LoggerManager
{
public:
    static Logger &get(const std::string &name);

private:
    LoggerManager() = default;
    ~LoggerManager() = default;

    // Use pointers to avoid static initialization/destruction issues
    static std::unordered_map<std::string, std::unique_ptr<Logger>> *m_loggers_ptr;
    static std::mutex *m_manager_mutex_ptr;
};

void
test_logging_levels(const std::string &logger_name);

inline void
init(py::module &m)
{
    auto m_logging = m.def_submodule("logging", "Internal functions for logging.");

    m_logging.def("test_logging_levels", &test_logging_levels,
                  "Test function to demonstrate logging at different levels",
                  py::arg("logger_name"));
}

}  // namespace xtgeo::logging

#endif  // XTGEO_LOGGING_HPP_
