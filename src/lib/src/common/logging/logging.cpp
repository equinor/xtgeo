#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <xtgeo/logging.hpp>

// Define static members outside the namespace
std::unordered_map<std::string, std::unique_ptr<xtgeo::logging::Logger>>
  *xtgeo::logging::LoggerManager::m_loggers_ptr = nullptr;
std::mutex *xtgeo::logging::LoggerManager::m_manager_mutex_ptr = nullptr;

namespace xtgeo::logging {

Logger &
LoggerManager::get(const std::string &name)
{
    // Rest of the implementation unchanged
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        // Initialize the logger map
        m_loggers_ptr = new std::unordered_map<std::string, std::unique_ptr<Logger>>();
        m_manager_mutex_ptr = new std::mutex();
    });

    std::lock_guard<std::mutex> lock(*m_manager_mutex_ptr);

    auto it = m_loggers_ptr->find(name);
    if (it == m_loggers_ptr->end()) {
        // Create the logger in-place and return a reference to it
        auto result = m_loggers_ptr->emplace(name, std::make_unique<Logger>(name));
        return *(result.first->second);
    }
    return *(it->second);
}

// this is for unit testing in python
void
test_logging_levels(const std::string &logger_name)
{
    auto &logger = LoggerManager::get(logger_name);

    // Log at different levels
    logger.debug("This is a debug message from C++");
    logger.info("This is an info message from C++");
    logger.warning("This is a warning message from C++");
    logger.error("This is an error message from C++");
    logger.critical("This is a critical message from C++");
}
}  // namespace xtgeo::logging
