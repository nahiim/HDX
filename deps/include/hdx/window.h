
#pragma once

#include <iostream>

#include <vulkan/vulkan.hpp>

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>


#include <hdx/app_state.h>
#include <hdx/input.h>


class Window
{
public:

	float dt, t1, t2 = 0;
	float start_time, u_time;

	char *title;

	unsigned int sdl_extension_count;
	std::vector<const char*> sdl_extensions;


	Window(const char* title, unsigned int width, unsigned int height) : m_title(title), m_width(width), m_height(height)
	{
		//initialize SDL
		SDL_Init(SDL_INIT_VIDEO);

		SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_VULKAN);

		window = SDL_CreateWindow(
			title,
			SDL_WINDOWPOS_UNDEFINED,
			SDL_WINDOWPOS_UNDEFINED,
			width, height,
			window_flags
		);
	}

	~Window()
	{
	    //Disable text input
	    SDL_StopTextInput();

	    //Destroy window
	    SDL_DestroyWindow( window );
	    window = NULL;

	    //Quit SDL subsystems
	    SDL_Quit();
	}



	void loop(AppState &app_state, float& delta_time)
	{
        t1 = SDL_GetTicks();
		delta_time = t1 - t2;
        t2 = t1;

        u_time = SDL_GetTicks() - start_time;

		// Update the input state
		Input::Update();
	}

	void getExtensions()
	{
		// Get the required extension count
		if (!SDL_Vulkan_GetInstanceExtensions(window, &sdl_extension_count, nullptr))
		{
			std::cout << "Failed to get SDL extensions\n";
		}

		sdl_extensions = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME }; // Sample additional extension 
		size_t additional_extension_count = sdl_extensions.size();
		sdl_extensions.resize(additional_extension_count + sdl_extension_count);

		if (!SDL_Vulkan_GetInstanceExtensions(window, &sdl_extension_count, sdl_extensions.data() + additional_extension_count))
		{
			std::cout << "Failed to get SDL extensions\n";
		}
	}



	void createSurface(vk::SurfaceKHR &surface, vk::Instance &instance)
	{
		VkSurfaceKHR c_style_surface; vk::SurfaceKHR;

		if (SDL_Vulkan_CreateSurface(window, instance, &c_style_surface) != SDL_TRUE)
		{
			std::cout << "Failed to abstract SDL surface for Vulkan\n";
		}

		//copy constructor converts to hpp convention
		surface = (vk::SurfaceKHR)c_style_surface;
	}

	void getFramebufferSize(int &width, int &height)
	{
		SDL_Vulkan_GetDrawableSize(window, &width, &height);
	}


	unsigned int getWidth()
	{
		return m_width;
	}

	unsigned int getHeight()
	{
		return m_height;
	}



private:

	SDL_Window *window;
	SDL_Event event;

	//Screen dimension constants
	unsigned int m_width;
	unsigned int m_height;

	const char* m_title;

};