
#include "Application.h"


Application::Application()
{
	window = new Window("SKY", WIDTH, HEIGHT);
	window->getExtensions();
	hdx::createInstance(instance, window, "SKYB", enable_validation_layers, validation_layers);
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
	camera3D = PerspectiveCamera(
		glm::vec3(0.0f, 0.0f, 5.0f), // eye
		glm::vec3(0.0f, 0.0f, 0.0f), // center
		glm::radians(45.0f), // fov
		aspectRatio, //aspect ratio
		0.1f, 1000.0f); // near and far points

	vp.projection = camera3D.getProjectionMatrix();
	vp.view = glm::mat4((camera3D.getViewMatrix()));
	model = glm::mat4(1.0f);

	light.color = glm::vec4(1.0f, 1.0f, 1.0f, 0.2f);
	light.position = glm::vec4(3.0f, 1.0f, 5.0f, 0.0f);

	instances = {
		{ glm::mat4(1.0f), glm::vec4(1.0, 0.765, 0.336, 1.0), 1.0, 0.3, 1.0 },
		{ glm::mat4(1.0f), glm::vec4(1.0, 0.765, 0.336, 1.0), 1.0, 0.3, 1.0 },
		{ glm::mat4(1.0f), glm::vec4(0.0, 0.0, 1.0, 1.0), 1.0, 1.0, 1.0 }
	};
	for (uint32_t i = 0; i < instance_count; i++)
	{
		hdx::translate(instances[i].modelMatrix, i * 3, 0, 0);
	}

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

	command_pool = hdx::createCommandPool(device, queue_family_index);
	command_buffer = hdx::allocateCommandBuffer(device, command_pool);
	s_command_buffer = hdx::allocateCommandBuffer(device, command_pool);

	renderpass = hdx::createRenderpass(device, msaa_samples, vk::Format::eR8G8B8A8Srgb);
	createImageDesc(device, color_image, vk::Format::eR8G8B8A8Srgb, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	createImageDesc(device, depth_image, vk::Format::eD32Sfloat, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	for (size_t i = 0; i < swapchain_size; i++)
	{
		swapchain_imageviews.push_back(hdx::createImageView(device, swapchain_images[i], vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, 1, 1, vk::ImageViewType::e2D));
		framebuffers.push_back(hdx::createFramebuffer(device, swapchain_imageviews[i], color_image.imageview, depth_image.imageview, renderpass, extent));
	}

	for (const auto& face : faces)
	{
		stbi_uc* pixels = stbi_load(face.c_str(), &image_width, &image_height, &image_channels, STBI_rgb_alpha);
		if (!pixels)
			throw std::runtime_error("Failed to load texture image!");

		uint64_t face_size = image_width * image_height * 4; // 4 bytes per pixel (RGBA)
		all_pixels.insert(all_pixels.end(), pixels, pixels + face_size);

		stbi_image_free(pixels);
	}

	createImageDesc(device, cube_texture, vk::Format::eR8G8B8A8Srgb, image_width, image_height, vk::SampleCountFlagBits::e1, cubemap_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_cube, 6, vk::ImageCreateFlagBits::eCubeCompatible, device_desc, 1);
	cube_sampler = hdx::createTextureSampler(device, device_desc.properties, 6);
	_DII = hdx::createDescriptorImageInfo(cube_texture, cube_sampler, vk::ImageLayout::eShaderReadOnlyOptimal);

	cube_tb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eTransferSrc, all_pixels.size());
	hdx::allocateBufferMemory(device, device_desc.memory_properties, cube_tb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, cube_tb, all_pixels.data(), all_pixels.size());

	cube_vb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, sizeof(float) * cube_coords.size());
	hdx::allocateBufferMemory(device, device_desc.memory_properties, cube_vb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, cube_vb, cube_coords.data(), sizeof(float) * cube_coords.size());

	ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(VPMatrices));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ub, &vp, sizeof(VPMatrices));
	_DBI = hdx::createDescriptorBufferInfo(ub, sizeof(VPMatrices));

	cube_binding_descriptions = { hdx::getBindingDescription(0, sizeof(float) * 4, vk::VertexInputRate::eVertex) };
	cube_attribute_descriptions.push_back(hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, 0));

	binding_descriptions = {
		hdx::getBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex),
		hdx::getBindingDescription(1, sizeof(InstanceData), vk::VertexInputRate::eInstance)
	};
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, position)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, normal)));
	for (int i = 0; i < 4; i++)
	{
		attribute_descriptions.push_back(hdx::getAttributeDescription(1, 4 + i, vk::Format::eR32G32B32A32Sfloat, offsetof(InstanceData, modelMatrix) + sizeof(glm::vec4) * i));
	}
	attribute_descriptions.push_back(hdx::getAttributeDescription(1, 8, vk::Format::eR32G32B32A32Sfloat, offsetof(InstanceData, albedo)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(1, 9, vk::Format::eR32Sfloat, offsetof(InstanceData, ao)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(1, 10, vk::Format::eR32Sfloat, offsetof(InstanceData, roughness)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(1, 11, vk::Format::eR32Sfloat, offsetof(InstanceData, metallic)));

	vertices.resize(vertex_count);
	indices.resize(index_count);
	hdx::fillGrid(positions, normals, uv, indices, tangents, stacks, slices);
	for (uint32_t i = 0; i < vertex_count; i++)
	{
		vertices[i].position = glm::vec4(positions[i], 1.0f);
		vertices[i].normal = glm::vec4(normals[i], 1.0f);
		vertices[i].uv = glm::vec4(uv[i], 1.0f, 1.0f);
		vertices[i].tangent = glm::vec4(tangents[i], 1.0f);
	}




	sphere_vb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, vertex_count * sizeof(Vertex));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, sphere_vb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, sphere_vb, vertices.data(), vertex_count * sizeof(Vertex));

	sphere_ib = hdx::createBuffer(device, vk::BufferUsageFlagBits::eIndexBuffer, sizeof(uint32_t) * index_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, sphere_ib, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, sphere_ib, indices.data(), sizeof(uint32_t) * index_count);

	light_ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Light));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, light_ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, light_ub, &light, sizeof(Light));
	_DBI_light = hdx::createDescriptorBufferInfo(light_ub, sizeof(Light));

	instance_b = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, sizeof(InstanceData) * instance_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, instance_b, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, instance_b, instances.data(), sizeof(InstanceData) * instance_count);

	hdx::beginSingleTimeCommands(device, s_command_buffer);
	hdx::transitionImageLayout(device, cube_texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, s_command_buffer, 1, 6);
	hdx::copyBufferToImage(device, cube_tb, cube_texture, image_width, image_height, 6, s_command_buffer);
	hdx::transitionImageLayout(device, cube_texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, s_command_buffer, 1, 6);
	hdx::endSingleTimeCommands(device, s_command_buffer, command_pool, queue);

	pool_sizes = {
		vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 4),
		vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 3)
	};
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, 2);

	_DSLB = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
	};

	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);

	_WDS = {
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI, 0),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, _DII, 1)
	};
	device.updateDescriptorSets(2, _WDS.data(), 0, nullptr);



	_DSLB0 = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
	};

	_DSL0 = hdx::createDescriptorSetLayout(device, _DSLB0);
	_DS0 = hdx::allocateDescriptorSet(device, _DSL0, descriptor_pool);

	_WDS = {
		hdx::createWriteDescriptorSet(_DS0, vk::DescriptorType::eUniformBuffer, _DBI, 0),
		hdx::createWriteDescriptorSet(_DS0, vk::DescriptorType::eUniformBuffer, _DBI_light, 1),
		hdx::createWriteDescriptorSet(_DS0, vk::DescriptorType::eCombinedImageSampler, _DII, 2)
	};
	device.updateDescriptorSets(3, _WDS.data(), 0, nullptr);

	pipeline = hdx::createGraphicsPipeline(device, pipeline_layout, renderpass, msaa_samples, "res/shaders/skybox.vert.spv", "res/shaders/skybox.frag.spv", cube_binding_descriptions, cube_attribute_descriptions, _DSL, vk::PrimitiveTopology::eTriangleList, extent);
	sphere_pipeline = hdx::createGraphicsPipeline(device, sphere_pl, renderpass, msaa_samples, "res/shaders/shader.vert.spv", "res/shaders/shader.frag.spv", binding_descriptions, attribute_descriptions, _DSL0, vk::PrimitiveTopology::eTriangleList, extent);

	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;
}


void Application::update(float delta_time, AppState& app_state)
{
	hdx::rotate(instances[1].modelMatrix, 0, 0.05, 0);
	hdx::copyToDevice(device, instance_b, instances.data(), sizeof(InstanceData) * instance_count);
	float rv = 0.1f * delta_time;
	float mv = 0.01f * delta_time;
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

	vp.view = glm::mat4((camera3D.getViewMatrix()));
	hdx::copyToDevice(device, ub, &vp, sizeof(VPMatrices));

	device.waitForFences({ in_flight_fence }, true, UINT64_MAX);
	device.resetFences({ in_flight_fence });

	uint32_t image_index;
	vk::Result acquire_result = device.acquireNextImageKHR(swap_chain, UINT64_MAX, image_available_semaphore, nullptr, &image_index);
	if (acquire_result == vk::Result::eErrorOutOfDateKHR)
	{
		hdx::recreateSwapChain(device, surface,
			vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eFifo, capabilities,
			WIDTH, HEIGHT, extent,
			swap_chain, color_image, depth_image, swapchain_images, swapchain_imageviews, framebuffers,
			vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, renderpass, device_desc);

		return;
	}
	device.resetFences({ in_flight_fence });
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources });


	vk::Buffer cube_vbs[] = { cube_vb.buffer }, sphere_vbs[] = { sphere_vb.buffer, instance_b.buffer };
	uint64_t offsets[] = { 0 };
	uint64_t offsets_i[] = { 0, 0 };

	std::vector<vk::ClearValue> clear_values = {
		vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
		vk::ClearDepthStencilValue(1.0f, 0)
	};

	hdx::beginRenderpass(command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
		hdx::recordCommandBuffer(pipeline, pipeline_layout, 36, command_buffer, cube_vbs, _DS, offsets);
		hdx::recordCommandBuffer(sphere_pipeline, sphere_pl, index_count, command_buffer, sphere_vbs, sphere_ib.buffer, _DS0, offsets_i, 2, instance_count);
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

	queue.waitIdle();
	device.waitIdle();
}



Application::~Application()
{
	cleanupSwapchain(device, swap_chain, swapchain_imageviews, framebuffers, color_image, depth_image);

	hdx::cleanupBuffer(device, cube_vb);
	hdx::cleanupBuffer(device, cube_tb);
	hdx::cleanupBuffer(device, ub);
	hdx::cleanupBuffer(device, sphere_vb);
	hdx::cleanupBuffer(device, sphere_ib);
	hdx::cleanupBuffer(device, light_ub);
	hdx::cleanupBuffer(device, instance_b);
	hdx::cleanupImage(device, cube_texture);
	device.destroySampler(cube_sampler);

	device.destroyPipeline(pipeline);
	device.destroyPipelineLayout(pipeline_layout);
	device.destroyPipeline(sphere_pipeline);
	device.destroyPipelineLayout(sphere_pl);
	device.destroyRenderPass(renderpass);

	device.destroyDescriptorPool(descriptor_pool);
	device.destroyDescriptorSetLayout(_DSL);
	device.destroyDescriptorSetLayout(_DSL0);

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