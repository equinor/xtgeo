#ifndef XTGEO_LOGGING_HPP_
#define XTGEO_LOGGING_HPP_
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iomanip>
#include <sstream>
#include <string>

namespace py = pybind11;

class Logger
{
public:
    explicit Logger(const std::string &name)
      : logger_name(name)
    {
        py::gil_scoped_acquire acquire;  // Acquire GIL (Global Interpreter Lock)
        py::object logging = py::module::import("logging");
        logger = logging.attr("getLogger")(logger_name);
    }

    template<typename... Args>
    void log(const std::string &level, const std::string &message, Args... args)
    {
        std::ostringstream oss;
        ((oss << args << " "),
         ...);  // Fold expression to concatenate arguments with spaces
        std::string formatted_message =
          "[C++: " + logger_name + "] " + message + oss.str();

        {
            py::gil_scoped_acquire acquire;  // Acquire GIL (Global Interpreter Lock)
            if (level == "debug") {
                logger.attr("debug")(formatted_message);
            } else if (level == "info") {
                logger.attr("info")(formatted_message);
            } else if (level == "warning") {
                logger.attr("warning")(formatted_message);
            } else if (level == "error") {
                logger.attr("error")(formatted_message);
            } else if (level == "critical") {
                logger.attr("critical")(formatted_message);
            } else {
                logger.attr("info")(formatted_message);
            }
        }  // GIL is released here
    }

    template<typename... Args>
    void debug(const std::string &message, Args... args)
    {
        log("debug", message, args...);
    }

    template<typename... Args>
    void info(const std::string &message, Args... args)
    {
        log("info", message, args...);
    }

    template<typename... Args>
    void warning(const std::string &message, Args... args)
    {
        log("warning", message, args...);
    }

    template<typename... Args>
    void error(const std::string &message, Args... args)
    {
        log("error", message, args...);
    }

    template<typename... Args>
    void critical(const std::string &message, Args... args)
    {
        log("critical", message, args...);
    }

private:
    std::string logger_name;
    py::object logger;
};

#endif  // XTGEO_LOGGING_HPP_
