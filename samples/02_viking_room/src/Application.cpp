
#include "Application.h"

void loadModel(const char* MODEL_PATH, std::vector<Vertex> &vertices, std::vector<uint32_t> &indices, uint64_t &sizeof_vertices, uint64_t &sizeof_indices)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;

	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH))
	{
		throw std::runtime_error(warn + err);
	}

	for (const auto& shape : shapes) {
		for (const auto& index : shape.mesh.indices) {
			Vertex vertex{};

			vertex.position = {
				attrib.vertices[3 * index.vertex_index + 0],
				attrib.vertices[3 * index.vertex_index + 1],
				attrib.vertices[3 * index.vertex_index + 2]
			};

			vertex.tex_coord = {
				attrib.texcoords[2 * index.texcoord_index + 0],
				1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
			};

			vertex.color = { 1.0f, 1.0f, 1.0f, 1.0f };

			auto it = std::find_if(vertices.begin(), vertices.end(),
				[&](const Vertex& v) { return v == vertex; });

			if (it == vertices.end()) {
				vertices.push_back(vertex);
				indices.push_back(static_cast<uint32_t>(vertices.size() - 1));
			}
			else {
				indices.push_back(static_cast<uint32_t>(std::distance(vertices.begin(), it)));
			}
		}
	}
	std::cout << "Size of Vertex: " << sizeof(Vertex) << std::endl;
	std::cout << "Size of uint32_t: " << sizeof(uint32_t) << std::endl;

	sizeof_vertices = sizeof(vertices[0]) * vertices.size();
	sizeof_indices = sizeof(indices[0]) * indices.size();
}


Application::Application()
{
	window = new Window("HYDROXY - Spheres", WIDTH, HEIGHT);
	window->getExtensions();
	hdx::createInstance(instance, window, "Spheres", enable_validation_layers, validation_layers);
	dldi = vk::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr);
	hdx::createDebugMessenger(debug_messenger, instance, dldi);
	window->createSurface(surface, instance);

	hdx::getPhysicalDevices(instance, device_descs);
	device_desc = device_descs[0];
	hdx::findQueueFamilies(device_desc);

	device = hdx::createLogicalDevice(device_desc, device_extensions, validation_layers, enable_validation_layers);

	for (uint32_t i = 0; i < device_desc.queue_families.size(); i++)
	{
		if (device_desc.queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics)
		{
			queue_family_index = i;
			queue = device.getQueue(queue_family_index, 0);
			break;  // Break the loop once a queue is found
		}
	}

	float aspectRatio = float(WIDTH) / float(HEIGHT);


	// Initialize the vp structure
	room_mvp.model = glm::mat4(1.0f);
//	.projection[1][1] *= -1;	

	camera3D = PerspectiveCamera(
		glm::vec3(0.0f, 0.0f, 5.0f), // eye
		glm::vec3(0.0f, 0.0f, 0.0f), // center
		glm::radians(45.0f), // fov
		aspectRatio, //aspect ratio
		0.1f, 1000.0f); // near and far points
	room_mvp.projection = camera3D.getProjectionMatrix();


	device_desc.physical_device.getMemoryProperties(&device_desc.memory_properties);
	device_desc.physical_device.getProperties(&device_desc.properties);
	device_desc.features.samplerAnisotropy = VK_TRUE;
	msaa_samples = hdx::getMaxUsableSampleCount(device_desc.physical_device);
	device_desc.physical_device.getFormatProperties(vk::Format::eR8G8B8A8Srgb, &device_desc.format_properties);

	capabilities = device_desc.physical_device.getSurfaceCapabilitiesKHR(surface);
	formats = device_desc.physical_device.getSurfaceFormatsKHR(surface);
	presentModes = device_desc.physical_device.getSurfacePresentModesKHR(surface);

	swap_chain = hdx::createSwapchain(device, surface, vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eImmediate, capabilities, WIDTH, HEIGHT, extent);
	swapchain_images = device.getSwapchainImagesKHR(swap_chain);
	swapchain_size = swapchain_images.size();

	renderpass = hdx::createRenderpass(device, msaa_samples, vk::Format::eR8G8B8A8Srgb);
	command_pool = hdx::createCommandPool(device, queue_family_index);
	command_buffer = hdx::allocateCommandBuffer(device, command_pool);
	s_command_buffer = hdx::allocateCommandBuffer(device, command_pool);

	room_VIBD = {hdx::getBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex)};
	room_VIAD.push_back(hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)));
	room_VIAD.push_back(hdx::getAttributeDescription(0, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, color)));
	room_VIAD.push_back(hdx::getAttributeDescription(0, 2, vk::Format::eR32G32Sfloat, offsetof(Vertex, tex_coord)));
	
	room_DPS.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, swapchain_size));
	room_DPS.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, swapchain_size));
	descriptor_pool = hdx::createDescriptorPool(device, room_DPS, swapchain_size);

	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load("res/textures/viking_room.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	room_TB_size = texWidth * texHeight * 4;

	if (!pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	room_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
	room_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment));
	room_DSL = hdx::createDescriptorSetLayout(device, room_DSLB);
	room_pipeline = hdx::createGraphicsPipeline(device, room_PL, renderpass, msaa_samples, "res/shaders/model.vert.spv", "res/shaders/model.frag.spv", room_VIBD, room_VIAD, room_DSL, vk::PrimitiveTopology::eTriangleList, extent);
	loadModel("res/models/viking_room.obj", room_vertices, room_indices, room_VB_size, room_IB_size);

	room_VB = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, room_VB_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, room_VB, host_VC);
	hdx::copyToDevice(device, room_VB, room_vertices.data(), room_VB_size);
	room_IB = hdx::createBuffer(device, vk::BufferUsageFlagBits::eIndexBuffer, room_IB_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, room_IB, host_VC);
	hdx::copyToDevice(device, room_IB, room_indices.data(), room_IB_size);
	room_TB = hdx::createBuffer(device, vk::BufferUsageFlagBits::eTransferSrc, room_TB_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, room_TB, host_VC);
	hdx::copyToDevice(device, room_TB, pixels, room_TB_size);			stbi_image_free(pixels);
	room_UB = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MVP));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, room_UB, host_VC);
	hdx::copyToDevice(device, room_UB, &room_mvp, sizeof(MVP));

	room_sampler = hdx::createTextureSampler(device, device_desc.properties, 1);
	hdx::createImageDesc(device, room_texture, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, vk::SampleCountFlagBits::e1, sampled_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_2d, 1, {}, device_desc, 1);

	hdx::beginSingleTimeCommands(device, s_command_buffer);
		hdx::transitionImageLayout(device, room_texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, s_command_buffer, 1, 1);
		hdx::copyBufferToImage(device, room_TB, room_texture, texWidth, texHeight, 1, s_command_buffer);
		hdx::transitionImageLayout(device, room_texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, s_command_buffer, 1, 1);
	hdx::endSingleTimeCommands(device, s_command_buffer, command_pool, queue);

	hdx::cleanupBuffer(device, room_TB);

	room_UB.size = sizeof(MVP);
	room_DS = hdx::allocateDescriptorSet(device, room_DSL, descriptor_pool);
	room_DBI = hdx::createDescriptorBufferInfo(room_UB, room_UB.size);
	room_DII = hdx::createDescriptorImageInfo(room_texture, room_sampler, vk::ImageLayout::eShaderReadOnlyOptimal);
	room_WDS.push_back(hdx::createWriteDescriptorSet(room_DS, vk::DescriptorType::eUniformBuffer, room_DBI, 0));
	room_WDS.push_back(hdx::createWriteDescriptorSet(room_DS, vk::DescriptorType::eCombinedImageSampler, room_DII, 1));
	device.updateDescriptorSets(2, room_WDS.data(), 0, nullptr);

	createImageDesc(device, color_image, vk::Format::eR8G8B8A8Srgb, WIDTH, HEIGHT, msaa_samples, color_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_2d, 1, {}, device_desc, 1);
	createImageDesc(device, depth_image, vk::Format::eD32Sfloat, WIDTH, HEIGHT, msaa_samples, depth_usage_flags, vk::ImageAspectFlagBits::eDepth, image_type_2d, view_type_2d, 1, {}, device_desc, 1);
	for (size_t i = 0; i < swapchain_size; i++)
	{
		swapchain_imageviews.push_back(hdx::createImageView(device, swapchain_images[i], vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, 1, 1, view_type_2d));
		framebuffers.push_back(hdx::createFramebuffer(device, swapchain_imageviews[i], color_image.imageview, depth_image.imageview, renderpass, extent));
	}

	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;
}


void Application::update(float delta_time, AppState& app_state)
{
//	std::cout << delta_time << " milliseconds" << std::endl;
	float rv = 0.02f;
	float mv = 0.002f;
	if (Input::GetKey(Input::KEY_I))
	{
		camera3D.rotate(-rv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_J))
	{
		camera3D.rotate(0, rv, 0);
	}
	if (Input::GetKey(Input::KEY_K))
	{
		camera3D.rotate(rv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_L))
	{
		camera3D.rotate(0, -rv, 0);
	}

	if (Input::GetKey(Input::KEY_UP))
	{
		hdx::rotate(room_mvp.model, 0.2f, 0.0f, 0.0f);
	}
	if (Input::GetKey(Input::KEY_DOWN))
	{
		hdx::rotate(room_mvp.model, -0.2f, 0.0f, 0.0f);
	}
	if (Input::GetKey(Input::KEY_LEFT))
	{
		hdx::rotateModel(room_mvp.model, 0.2f, 'y');
	}
	if (Input::GetKey(Input::KEY_RIGHT))
	{
		hdx::rotateModel(room_mvp.model, -0.2f, 'y');
	}

	if (Input::GetKey(Input::KEY_W))
	{
		camera3D.translate(0, 0, mv);
	}
	if (Input::GetKey(Input::KEY_A))
	{
		camera3D.translate(-mv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_S))
	{
		camera3D.translate(0, 0, -mv);
	}
	if (Input::GetKey(Input::KEY_D))
	{
		camera3D.translate(mv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_ESCAPE))
	{
		app_state.running = false;
	}
	room_mvp.view = camera3D.getViewMatrix();
	
	hdx::copyToDevice(device, room_UB, &room_mvp, room_UB.size);


	device.waitForFences({ in_flight_fence }, true, UINT64_MAX);
	device.resetFences({ in_flight_fence });

	vk::Result acquire_result = device.acquireNextImageKHR(swap_chain, UINT64_MAX, image_available_semaphore, nullptr, &image_index);
	if (acquire_result == vk::Result::eErrorOutOfDateKHR)
	{
		recreateSwapChain(device, surface,
			vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eImmediate, capabilities,
			WIDTH, HEIGHT, extent,
			swap_chain, color_image, depth_image, swapchain_images, swapchain_imageviews, framebuffers,
			vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, renderpass, device_desc);

		return;
	}

	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources });
	std::vector<vk::ClearValue> clear_values = {
	vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
	vk::ClearDepthStencilValue(1.0f, 0)
	};
	vk::Buffer room_vbs[] = { room_VB.buffer };
	uint64_t offsets[] = { 0 };
	hdx::beginRenderpass(command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
		hdx::recordCommandBuffer(room_pipeline, room_PL, room_indices.size(), command_buffer, room_vbs, room_IB.buffer, room_DS, offsets, 1, 1);
	hdx::endRenderpass(command_buffer);

	vk::SubmitInfo submit_info{};
	submit_info = 0;
	submit_info.sType = vk::StructureType::eSubmitInfo;
	submit_info.pWaitDstStageMask = wait_stages;
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = &image_available_semaphore;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &render_finished_semaphore;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffer;
	queue.submit(submit_info, in_flight_fence);

	vk::PresentInfoKHR present_info;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &render_finished_semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &swap_chain;
	present_info.pImageIndices = &image_index;

	vk::Result present_result;
	try
	{
		present_result = queue.presentKHR(present_info);
	}
	catch (vk::OutOfDateKHRError error)
	{
		recreateSwapChain(device, surface,
			vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eImmediate, capabilities,
			WIDTH, HEIGHT, extent,
			swap_chain, color_image, depth_image, swapchain_images, swapchain_imageviews, framebuffers,
			vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, renderpass, device_desc);

		return;
	}

	current_frame++;
	current_frame = (current_frame == swapchain_size) ? 0 : current_frame;
}


Application::~Application()
{
	device.waitIdle();

	hdx::cleanupSwapchain(device, swap_chain, swapchain_imageviews, framebuffers, color_image, depth_image);

	cleanupImage(device, room_texture);
	hdx::cleanupBuffer(device, room_UB);
	hdx::cleanupBuffer(device, room_VB);
	hdx::cleanupBuffer(device, room_IB);
	device.destroySampler(room_sampler);

	device.destroyDescriptorPool(descriptor_pool);

	device.destroyDescriptorSetLayout(room_DSL);
	device.destroyPipeline(room_pipeline);
	device.destroyPipelineLayout(room_PL);

	device.destroyRenderPass(renderpass);

	device.destroySemaphore(render_finished_semaphore);
	device.destroySemaphore(image_available_semaphore);
	device.destroyFence(in_flight_fence);

	device.destroyCommandPool(command_pool);

	device.destroy();
	instance.destroySurfaceKHR(surface);
	if (enable_validation_layers) { instance.destroyDebugUtilsMessengerEXT(debug_messenger, nullptr, dldi); }

	instance.destroy();	std::cout << "Application DESTROYED!!\n";

	delete window;
}