
#include "Application.h"

void computeTangents(std::vector<glm::vec3>& tangents, glm::vec2* texCoords, uint32_t quadVertexCount)
{
	for (int i = 0; i < quadVertexCount; i += 3)
	{
		// Shortcuts for UVs
		glm::vec2 uv0 = texCoords[i + 0];
		glm::vec2 uv1 = texCoords[i + 1];
		glm::vec2 uv2 = texCoords[i + 2];

		// UV delta
		glm::vec2 deltaUV1 = uv1 - uv0;
		glm::vec2 deltaUV2 = uv2 - uv0;

		glm::vec3 tangent = glm::vec3(deltaUV1.x, deltaUV1.y, 0.0f);

		// Set the same tangent for all three vertices of the triangle.
		tangents.push_back(tangent);
		tangents.push_back(tangent);
		tangents.push_back(tangent);
	}
}
void fillGrid(
	glm::vec3* positionPtr, glm::vec2* texCoordPtr, glm::vec3* normalPtr,
	uint32_t* indexPtr, glm::vec3 upVector, uint32_t n,
	std::vector<glm::vec3> tangents, glm::vec2* texCoords, uint32_t quadVertexCount)
{
	for (unsigned int x = 0; x < n; x++) for (unsigned int z = 0; z < n; z++)
	{
		*positionPtr++ = glm::vec3((z / (float)n) - 0.5f, 0, (x / (float)n) - 0.5f);

		*texCoordPtr++ = glm::vec2((z / (float)n) * n, (x / (float)n) * n);

		*normalPtr++ = upVector;
	}

	for (unsigned int x = 0; x < n - 1; x++) for (unsigned int z = 0; z < n - 1; z++)
	{
		*indexPtr++ = (x * n) + z;
		*indexPtr++ = ((x + 1) * n) + z;
		*indexPtr++ = ((x + 1) * n) + z + 1;

		*indexPtr++ = (x * n) + z;
		*indexPtr++ = ((x + 1) * n) + z + 1;
		*indexPtr++ = (x * n) + z + 1;
	}

	computeTangents(tangents, texCoords, quadVertexCount);
}

Application::Application()
{
	float aspectRatio = float(WIDTH) / float(HEIGHT);
	camera3D = PerspectiveCamera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
	mvp.projection = camera3D.getProjectionMatrix();
	mvp.view = camera3D.getViewMatrix();
	mvp.model = glm::mat4(1.0f);

	window = new Window("HYDROXY - Particles(Compute)", WIDTH, HEIGHT);
	window->getExtensions();
	hdx::createInstance(instance, window, "Particles", enable_validation_layers, validation_layers);
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

	device_desc.physical_device.getMemoryProperties(&device_desc.memory_properties);
	device_desc.physical_device.getProperties(&device_desc.properties);
	device_desc.features.samplerAnisotropy = VK_TRUE;
	msaa_samples = hdx::getMaxUsableSampleCount(device_desc.physical_device);
	device_desc.physical_device.getFormatProperties(vk::Format::eR8G8B8A8Srgb, &device_desc.format_properties);
	limits = device_desc.properties.limits;
	std::cout << "maxComputeWorkGroupCount: "
		<< limits.maxComputeWorkGroupCount[0] << ", "
		<< limits.maxComputeWorkGroupCount[1] << ", "
		<< limits.maxComputeWorkGroupCount[2] << std::endl;

	std::cout << "maxComputeWorkGroupInvocations: "
		<< limits.maxComputeWorkGroupInvocations << std::endl;

	std::cout << "maxComputeWorkGroupSize: "
		<< limits.maxComputeWorkGroupSize[0] << ", "
		<< limits.maxComputeWorkGroupSize[1] << ", "
		<< limits.maxComputeWorkGroupSize[2] << std::endl;
	device_desc.features.samplerAnisotropy = VK_TRUE;

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

	binding_descriptions = { hdx::getBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex) };
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0));
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 1, vk::Format::eR32G32Sfloat, offsetof(Vertex, texcoord)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 2, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, normal)));
	
	_DPS.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, swapchain_size));
	_DPS.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, swapchain_size));
	descriptor_pool = hdx::createDescriptorPool(device, _DPS, 1);
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment));
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);

	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load("res/textures/vulk.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	uint64_t texture_size = texWidth * texHeight * 4;
	if (!pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	pipeline = hdx::createGraphicsPipeline(device, pipeline_layout, renderpass, msaa_samples, "res/shaders/shader.vert.spv", "res/shaders/shader.frag.spv", binding_descriptions, attribute_descriptions, _DSL, vk::PrimitiveTopology::eTriangleList, extent);

	fillGrid(positionPtr, texCoordPtr, normalPtr, indexPtr, upVector, n, tangents, texCoords, vertex_count);
	vertices.resize(vertex_count);
	for (size_t i = 0; i < vertex_count; i++)
	{
		vertices[i].position = positions[i];
		vertices[i].texcoord = texCoords[i] / static_cast<float>(n);
		vertices[i].normal = normals[i];
	}

	vb_desc = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, sizeof(Vertex) * vertex_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, vb_desc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, vb_desc, vertices.data(), sizeof(Vertex) * vertex_count);
	ib_desc = hdx::createBuffer(device, vk::BufferUsageFlagBits::eIndexBuffer, sizeof(uint32_t) * index_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ib_desc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ib_desc, indices, sizeof(uint32_t) * index_count);
	ub_desc = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MVP));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ub_desc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ub_desc, &mvp, sizeof(MVP));
	tb_desc = hdx::createBuffer(device, vk::BufferUsageFlagBits::eTransferSrc, texture_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, tb_desc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, tb_desc, pixels, texture_size);			stbi_image_free(pixels);

	sampler = hdx::createTextureSampler(device, device_desc.properties, 1);
	hdx::createImageDesc(device, texture, vk::Format::eR8G8B8A8Srgb, texWidth, texHeight, vk::SampleCountFlagBits::e1, sampled_usage_flags, vk::ImageAspectFlagBits::eColor, image_type_2d, view_type_2d, 1, {}, device_desc, 1);

	hdx::beginSingleTimeCommands(device, s_command_buffer);
		hdx::transitionImageLayout(device, texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, s_command_buffer, 1, 1);
		hdx::copyBufferToImage(device, tb_desc, texture, texWidth, texHeight, 1, s_command_buffer);
		hdx::transitionImageLayout(device, texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, s_command_buffer, 1, 1);
	hdx::endSingleTimeCommands(device, s_command_buffer, command_pool, queue);

	hdx::cleanupBuffer(device, tb_desc);

	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);
	_DBI = hdx::createDescriptorBufferInfo(ub_desc, sizeof(MVP));
	_DII = hdx::createDescriptorImageInfo(texture, sampler, vk::ImageLayout::eShaderReadOnlyOptimal);
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI, 0));
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, _DII, 1));
	device.updateDescriptorSets(2, _WDS.data(), 0, nullptr);

	createImageDesc(device, color_image, vk::Format::eR8G8B8A8Srgb, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	createImageDesc(device, depth_image, vk::Format::eD32Sfloat, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	for (size_t i = 0; i < swapchain_size; i++)
	{
		swapchain_imageviews.push_back(hdx::createImageView(device, swapchain_images[i], vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, 1, 1, vk::ImageViewType::e2D));
		framebuffers.push_back(hdx::createFramebuffer(device, swapchain_imageviews[i], color_image.imageview, depth_image.imageview, renderpass, extent));
	}
	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	hdx::rotate(mvp.model, 90, 0, 0);
}


void Application::update(float delta_time, AppState& app_state)
{
	mvp.view = camera3D.getViewMatrix();
	hdx::copyToDevice(device, ub_desc, &mvp, sizeof(MVP));
	float rv = 0.02f * delta_time;
	float mv = 0.002f * delta_time;
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

	device.waitForFences({ in_flight_fence }, true, UINT64_MAX);
	device.resetFences({ in_flight_fence });

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
	command_buffer.reset({ vk::CommandBufferResetFlagBits::eReleaseResources });

	std::vector<vk::ClearValue> clear_values = {
		vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
		vk::ClearDepthStencilValue(1.0f, 0)
	};
	vk::Buffer vbs[] = { vb_desc.buffer };
	uint64_t offsets[] = { 0 };
	hdx::beginRenderpass(command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
		hdx::recordCommandBuffer(pipeline, pipeline_layout, index_count, command_buffer, vbs, ib_desc.buffer, _DS, offsets, 1, 1);
	hdx::endRenderpass(command_buffer);

	vk::SubmitInfo submit_info{};
	submit_info = 0;
	submit_info.sType = vk::StructureType::eSubmitInfo;
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = &image_available_semaphore;
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = &render_finished_semaphore;
	submit_info.pWaitDstStageMask = wait_stages;
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffer;
	queue.submit({ submit_info }, in_flight_fence);

	vk::PresentInfoKHR present_info;
	present_info.sType = vk::StructureType::ePresentInfoKHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &render_finished_semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = &swap_chain;
	present_info.pImageIndices = &image_index; // Use the acquired image index

	vk::Result present_result;
	try
	{
		present_result = queue.presentKHR(present_info);
	}
	catch (vk::OutOfDateKHRError error)
	{
		recreateSwapChain(device, surface,
			vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eFifo, capabilities,
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
	cleanupSwapchain(device, swap_chain, swapchain_imageviews, framebuffers, color_image, depth_image);

	hdx::cleanupBuffer(device, vb_desc);
	hdx::cleanupBuffer(device, ub_desc);
	hdx::cleanupBuffer(device, ib_desc);
	hdx::cleanupImage(device, texture);
	device.destroySampler(sampler);

	device.destroyPipeline(pipeline);
	device.destroyPipelineLayout(pipeline_layout);

	device.destroyRenderPass(renderpass);

	device.destroyDescriptorPool(descriptor_pool);
	device.destroyDescriptorSetLayout(_DSL);

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