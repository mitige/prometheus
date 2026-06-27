cmake_minimum_required (VERSION 3.18)

project("GigaLearnBot")

option(PROMETHEUS_BUILD_GUI "Build the Prometheus ImGui learner frontend" ON)
option(GGL_ENABLE_ROCKETSIM_CUDA "Build and link the RocketSimCuda physics backend" ON)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include RocketSimCuda before GigaLearnCPP so RLGymCPP can detect the target.
if(GGL_ENABLE_ROCKETSIM_CUDA)
	add_subdirectory(RocketSimCuda)
endif()

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)

# Include RLBot
add_subdirectory(RLBotCPP)

add_executable(GigaLearnBot
	"src/ExampleMain.cpp"
	"src/RLBotClient.cpp"
	"src/RLBotClient.h"
)
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)
target_link_libraries(GigaLearnBot GigaLearnCPP RLBotCPP)

if(PROMETHEUS_BUILD_GUI)
	include(FetchContent)

	set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
	FetchContent_Declare(glfw
		GIT_REPOSITORY https://github.com/glfw/glfw.git
		GIT_TAG        3.4
	)
	FetchContent_MakeAvailable(glfw)

	FetchContent_Declare(imgui
		GIT_REPOSITORY https://github.com/ocornut/imgui.git
		GIT_TAG        v1.91.8-docking
	)
	FetchContent_MakeAvailable(imgui)

	set(IMGUI_DIR ${imgui_SOURCE_DIR})
	add_library(imgui_lib STATIC
		${IMGUI_DIR}/imgui.cpp
		${IMGUI_DIR}/imgui_demo.cpp
		${IMGUI_DIR}/imgui_draw.cpp
		${IMGUI_DIR}/imgui_tables.cpp
		${IMGUI_DIR}/imgui_widgets.cpp
		${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
		${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
	)
	target_include_directories(imgui_lib PUBLIC
		${IMGUI_DIR}
		${IMGUI_DIR}/backends
	)
	target_link_libraries(imgui_lib PUBLIC glfw)
	if(WIN32)
		target_link_libraries(imgui_lib PUBLIC opengl32)
	else()
		find_package(OpenGL REQUIRED)
		target_link_libraries(imgui_lib PUBLIC OpenGL::GL)
	endif()

	add_executable(Prometheus
		"src/gui/GuiMain.cpp"
		"src/gui/GuiApp.cpp"
		"src/gui/GuiApp.h"
	)
	set_target_properties(Prometheus PROPERTIES LINKER_LANGUAGE CXX)
	set_target_properties(Prometheus PROPERTIES CXX_STANDARD 20)
	target_include_directories(Prometheus PRIVATE "src/gui")
	target_link_libraries(Prometheus PRIVATE imgui_lib GigaLearnCPP)
	target_compile_definitions(Prometheus PRIVATE IMGUI_IMPL_OPENGL_LOADER_CUSTOM)
endif()
