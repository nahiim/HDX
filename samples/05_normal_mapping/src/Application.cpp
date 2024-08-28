
#include "Application.h"


void computeTangents(uint32_t vert_count, std::vector<glm::vec2> uv, std::vector<glm::vec3>& tangents)
{
	for (int i = 0; i < vert_count; i += 3)
	{
		// Shortcuts for UVs
		glm::vec2 uv0 = uv[i + 0];
		glm::vec2 uv1 = uv[i + 1];
		glm::vec2 uv2 = uv[i + 2];

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

void fillGrid(std::vector<glm::vec3>& positions, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uv, std::vector<uint32_t>& indices, std::vector<glm::vec3>& tangents, uint32_t stacks, uint32_t slices)
{
	positions.clear();
	normals.clear();
	uv.clear();
	indices.clear();
	tangents.clear();

	// Generate positions, normals, and UVs
	for (uint32_t stack = 0; stack <= stacks; ++stack)
	{
		float stackAngle = glm::pi<float>() / stacks * stack;
		float sinStack = std::sin(stackAngle);
		float cosStack = std::cos(stackAngle);

		for (uint32_t slice = 0; slice <= slices; ++slice)
		{
			float sliceAngle = 2.0f * glm::pi<float>() / slices * slice;
			float sinSlice = std::sin(sliceAngle);
			float cosSlice = std::cos(sliceAngle);

			glm::vec3 position(cosSlice * sinStack, cosStack, sinSlice * sinStack);
			glm::vec3 normal = glm::normalize(position);
			glm::vec2 texCoord(static_cast<float>(slice) / slices, static_cast<float>(stack) / stacks);

			positions.push_back(position);
			normals.push_back(normal);
			uv.push_back(texCoord);
		}
	}

	// Generate indices
	for (uint32_t stack = 0; stack < stacks; ++stack)
	{
		for (uint32_t slice = 0; slice < slices; ++slice)
		{
			uint32_t first = (stack * (slices + 1)) + slice;
			uint32_t second = first + slices + 1;

			// Avoid creating indices for the last row of vertices
			if (stack < stacks - 1)
			{
				indices.push_back(first);
				indices.push_back(second);
				indices.push_back(first + 1);

				indices.push_back(second);
				indices.push_back(second + 1);
				indices.push_back(first + 1);
			}
		}
	}
	
	uint32_t vert_count = (slices + 1) * (stacks + 1);
	computeTangents(vert_count, uv, tangents);
}

Application::Application()
{
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
	float aspectRatio = float(WIDTH) / float(HEIGHT);
	camera3D = PerspectiveCamera(glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
	mvp.projection = camera3D.getProjectionMatrix();
	mvp.view = camera3D.getViewMatrix();
	mvp.model = glm::mat4(1.0f);
	mvp.view_pos = glm::vec4(camera3D.getPosition(), 1.0f);

	light.color = glm::vec4(1.0f);
	light.position = glm::vec4(0.0f, 0.0f, 5.0f, 1.0f);

	device_desc.physical_device.getMemoryProperties(&device_desc.memory_properties);
	device_desc.physical_device.getProperties(&device_desc.properties);
	device_desc.features.samplerAnisotropy = VK_TRUE;
	msaa_samples = hdx::getMaxUsableSampleCount(device_desc.physical_device);
	device_desc.physical_device.getFormatProperties(vk::Format::eR8G8B8A8Srgb, &device_desc.format_properties);
	device_desc.features.samplerAnisotropy = VK_TRUE;

	capabilities = device_desc.physical_device.getSurfaceCapabilitiesKHR(surface);
	formats = device_desc.physical_device.getSurfaceFormatsKHR(surface);
	presentModes = device_desc.physical_device.getSurfacePresentModesKHR(surface);

	swap_chain = hdx::createSwapchain(device, surface, vk::Format::eR8G8B8A8Srgb, vk::PresentModeKHR::eImmediate, capabilities, WIDTH, HEIGHT, extent);
	swapchain_images = device.getSwapchainImagesKHR(swap_chain);
	swapchain_size = swapchain_images.size();

	command_pool = hdx::createCommandPool(device, queue_family_index);
	command_buffer = hdx::allocateCommandBuffer(device, command_pool);
	s_cmd = hdx::allocateCommandBuffer(device, command_pool);

	renderpass = hdx::createRenderpass(device, msaa_samples, vk::Format::eR8G8B8A8Srgb);
	createImageDesc(device, color_image, vk::Format::eR8G8B8A8Srgb, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	createImageDesc(device, depth_image, vk::Format::eD32Sfloat, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	for (size_t i = 0; i < swapchain_size; i++)
	{
		swapchain_imageviews.push_back(hdx::createImageView(device, swapchain_images[i], vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, 1, 1, vk::ImageViewType::e2D));
		framebuffers.push_back(hdx::createFramebuffer(device, swapchain_imageviews[i], color_image.imageview, depth_image.imageview, renderpass, extent));
	}
	
	binding_descriptions = { hdx::getBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex) };
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, position)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, normal)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 2, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, uv)));
	attribute_descriptions.push_back(hdx::getAttributeDescription(0, 3, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent)));

	vertices.resize(vertex_count);
	indices.resize(index_count);
	fillGrid(positions, normals, uv, indices, tangents, stacks, slices);
	for (uint32_t i = 0; i < vertex_count; i++)
	{
		vertices[i].position = glm::vec4(positions[i], 1.0f);
		vertices[i].normal = glm::vec4(normals[i], 1.0f);
		vertices[i].uv = glm::vec4(uv[i], 1.0f, 1.0f);
		vertices[i].tangent = glm::vec4(tangents[i], 1.0f);
	}

	diffuse_width, diffuse_height, diffuse_channel;
	stbi_uc* diffuse_pixels = stbi_load("res/textures/diffuse.jpg", &diffuse_width, &diffuse_height, &diffuse_channel, STBI_rgb_alpha);
	diffuse_size = diffuse_width * diffuse_height * 4;
	if (!diffuse_pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	normal_width, normal_height, normal_channel;
	stbi_uc* normal_pixels = stbi_load("res/textures/normal.jpg", &normal_width, &normal_height, &normal_channel, STBI_rgb_alpha);
	normal_size = normal_width* normal_height * 4;
	if (!normal_pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	diffuse_sampler = hdx::createTextureSampler(device, device_desc.properties, 1);
	hdx::createImageDesc(device, diffuse_image_desc, vk::Format::eR8G8B8A8Srgb, diffuse_width, diffuse_height, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	normal_sampler = hdx::createTextureSampler(device, device_desc.properties, 1);
	hdx::createImageDesc(device, normal_image_desc, vk::Format::eR8G8B8A8Srgb, normal_width, normal_height, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);

	
	vb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, vertex_count * sizeof(Vertex));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, vb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, vb, vertices.data(), vertex_count * sizeof(Vertex));

	ib = hdx::createBuffer(device, vk::BufferUsageFlagBits::eIndexBuffer, sizeof(uint32_t) * index_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ib, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ib, indices.data(), sizeof(uint32_t) * index_count);

	diffuse_tb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eTransferSrc, diffuse_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, diffuse_tb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, diffuse_tb, diffuse_pixels, diffuse_size);	stbi_image_free(diffuse_pixels);

	normal_tb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eTransferSrc, normal_size);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, normal_tb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, normal_tb, normal_pixels, normal_size);	stbi_image_free(normal_pixels);

	ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MVP));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ub, &mvp, sizeof(MVP));

	light_ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Light));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, light_ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, light_ub, &light, sizeof(Light));

	hdx::beginSingleTimeCommands(device, s_cmd);
		hdx::transitionImageLayout(device, diffuse_image_desc, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, s_cmd, 1, 1);
		hdx::copyBufferToImage(device, diffuse_tb, diffuse_image_desc, diffuse_width, diffuse_height, 1, s_cmd);
		hdx::transitionImageLayout(device, diffuse_image_desc, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, s_cmd, 1, 1);

		hdx::transitionImageLayout(device, normal_image_desc, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, s_cmd, 1, 1);
		hdx::copyBufferToImage(device, normal_tb, normal_image_desc, normal_width, normal_height, 1, s_cmd);
		hdx::transitionImageLayout(device, normal_image_desc, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, s_cmd, 1, 1);
	hdx::endSingleTimeCommands(device, s_cmd, command_pool, queue);


	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex));
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment));
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment));
	_DSLB.push_back(hdx::createDescriptorSetLayoutBinding(3, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment));
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);

	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, swapchain_size));
	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, swapchain_size));
	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, swapchain_size));
	pool_sizes.push_back(hdx::createDescriptorPoolSize(vk::DescriptorType::eUniformBuffer, swapchain_size));
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, 1);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);

	_DBI_u = hdx::createDescriptorBufferInfo(ub, sizeof(MVP));
	_DII_diffuse = hdx::createDescriptorImageInfo(diffuse_image_desc, diffuse_sampler, vk::ImageLayout::eShaderReadOnlyOptimal);
	_DII_normal= hdx::createDescriptorImageInfo(normal_image_desc, normal_sampler, vk::ImageLayout::eShaderReadOnlyOptimal);
	_DBI_light = hdx::createDescriptorBufferInfo(light_ub, sizeof(Light));

	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI_u, 0));
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, _DII_diffuse, 1));
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, _DII_normal, 2));
	_WDS.push_back(hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI_light, 3));

	device.updateDescriptorSets(4, _WDS.data(), 0, nullptr);

	pipeline = hdx::createGraphicsPipeline(device, pipeline_layout, renderpass, msaa_samples, "res/shaders/shader.vert.spv", "res/shaders/shader.frag.spv", binding_descriptions, attribute_descriptions, _DSL, vk::PrimitiveTopology::eTriangleList, extent);

	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;
}


void Application::update(float delta_time, AppState& app_state)
{
//	hdx::rotate(mvp.model, 0, 2, 0);
	mvp.view = camera3D.getViewMatrix();
	hdx::copyToDevice(device, ub, &mvp, sizeof(MVP));
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

	vk::Buffer vbs[] = { vb.buffer };
	uint64_t offsets[] = { 0 };

	std::vector<vk::ClearValue> clear_values = {
		vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
		vk::ClearDepthStencilValue(1.0f, 0)
	};
	hdx::beginRenderpass(command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
		hdx::recordCommandBuffer(pipeline, pipeline_layout, index_count, command_buffer, vbs, ib.buffer, _DS, offsets, 1, 1);
	hdx::endRenderpass(command_buffer);

	vk::SubmitInfo submit_info{};
	submit_info.sType = vk::StructureType::eSubmitInfo;
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

	vk::SwapchainKHR swapChains[] = { swap_chain };
	vk::PresentInfoKHR present_info;
	present_info.sType = vk::StructureType::ePresentInfoKHR;
	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = &render_finished_semaphore;
	present_info.swapchainCount = 1;
	present_info.pSwapchains = swapChains;
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

	current_frame = (current_frame + 1) % swapchain_size;

	queue.waitIdle();
	device.waitIdle();
}



Application::~Application()
{
	cleanupSwapchain(device, swap_chain, swapchain_imageviews, framebuffers, color_image, depth_image);

	hdx::cleanupBuffer(device, vb);
	hdx::cleanupBuffer(device, ub);
	hdx::cleanupBuffer(device, light_ub);
	hdx::cleanupBuffer(device, ib);
	hdx::cleanupBuffer(device, diffuse_tb);
	hdx::cleanupBuffer(device, normal_tb);
	hdx::cleanupImage(device, diffuse_image_desc);
	hdx::cleanupImage(device, normal_image_desc);
	device.destroySampler(diffuse_sampler);
	device.destroySampler(normal_sampler);

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