
#include "Application.h"


// Function to generate a plane grid with indices
void generatePlaneGrid(int gridSize, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
	// Precompute the normal (pointing up)
	glm::vec4 normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);

	// Precompute the tangent (pointing along the X axis)
	glm::vec4 tangent = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);

	float step = 2.0f / gridSize; // Calculate step size based on gridSize

	// Generate vertices
	for (int z = 0; z <= gridSize; ++z) {
		for (int x = 0; x <= gridSize; ++x) {
			Vertex vertex;

			// Calculate position, mapping (x, z) to the range [-1, 1]
			vertex.position = glm::vec4(x * step - 1.0f, 0.0f, z * step - 1.0f, 1.0f);

			// Calculate UV coordinates, storing in a vec4 with z and w components as 0
			vertex.uv = glm::vec4(static_cast<float>(x) / gridSize, static_cast<float>(z) / gridSize, 0.0f, 0.0f);

			// Set the normal and tangent
			vertex.normal = normal;
			vertex.tangent = tangent;

			vertices.push_back(vertex);
		}
	}

	// Generate indices
	for (int z = 0; z < gridSize; ++z) {
		for (int x = 0; x < gridSize; ++x) {
			// Indices for the two triangles that make up each grid square
			uint32_t topLeft = z * (gridSize + 1) + x;
			uint32_t topRight = topLeft + 1;
			uint32_t bottomLeft = (z + 1) * (gridSize + 1) + x;
			uint32_t bottomRight = bottomLeft + 1;

			// Triangle 1
			indices.push_back(topLeft);
			indices.push_back(bottomLeft);
			indices.push_back(topRight);

			// Triangle 2
			indices.push_back(topRight);
			indices.push_back(bottomLeft);
			indices.push_back(bottomRight);
		}
	}
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
	plane_mvp.view = camera3D.getViewMatrix();
	plane_mvp.model = glm::mat4(1.0f);
	plane_mvp.view_pos = glm::vec4(camera3D.getPosition(), 1.0f);
	plane_mvp.projection = camera3D.getProjectionMatrix();
	plane_mvp.projection[1][1] *= -1;
	mvp.projection[1][1] *= -1;


	light.color = glm::vec4(1.0f);
	light.position = glm::vec4(0.0f, 10.0f, 0.0f, 1.0f);

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
	sh_cmd = hdx::allocateCommandBuffer(device, command_pool);

	renderpass = hdx::createRenderpass(device, msaa_samples, vk::Format::eR8G8B8A8Srgb);
	createImageDesc(device, color_image, vk::Format::eR8G8B8A8Srgb, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	createImageDesc(device, depth_image, vk::Format::eD32Sfloat, WIDTH, HEIGHT, msaa_samples, vk::ImageUsageFlagBits::eTransientAttachment | vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	for (size_t i = 0; i < swapchain_size; i++)
	{
		swapchain_imageviews.push_back(hdx::createImageView(device, swapchain_images[i], vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor, 1, 1, vk::ImageViewType::e2D));
		framebuffers.push_back(hdx::createFramebuffer(device, swapchain_imageviews[i], color_image.imageview, depth_image.imageview, renderpass, extent));
	}

	shadow_rp = hdx::createDepthRenderpass(device);
	createImageDesc(device, shadow_map, vk::Format::eD32Sfloat, 1024, 1024, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	shadow_fb = hdx::createFramebuffer(device, shadow_map.imageview, shadow_rp, 1024, 1024);
	ssampler = hdx::createShadowSampler(device, device_desc.properties);
	_DII_shadow = hdx::createDescriptorImageInfo(shadow_map, ssampler, vk::ImageLayout::eDepthStencilReadOnlyOptimal);

	glm::mat4 light_projection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, 1.0f, 7.5f);
	glm::mat4 light_view = glm::lookAt(glm::vec3(light.position), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	light_projection[1][1] = -1;
	light.mvp = { glm::mat4(1.0f), light_view, light_projection };

	binding_descriptions = { hdx::getBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex) };
	attribute_descriptions = {
		hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, position)),
		hdx::getAttributeDescription(0, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, normal)),
		hdx::getAttributeDescription(0, 2, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, uv)),
		hdx::getAttributeDescription(0, 3, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, tangent))
	};
/*
	hdx::fillPlaneGrid(positionPtr, texCoordPtr, normalPtr, indexPtr, upVector, n, plane_tangents, plane_positions, plane_texCoords, plane_vertex_count);
	plane_vertices.resize(plane_vertex_count);
	std::cout << "sdsdsds\n" << plane_index_count << "\n";
	for (size_t i = 0; i < plane_vertex_count; i++)
	{
		plane_vertices[i].position = glm::vec4(plane_positions[i], 1.0f);
		plane_vertices[i].uv = glm::vec4(plane_texCoords[i] / static_cast<float>(n), 0.0f, 0.0f);
		plane_vertices[i].normal = glm::vec4(plane_normals[i], 0.0f);
		plane_vertices[i].tangent = glm::vec4(plane_tangents[i], 0.0f);
	}
	*/

	generatePlaneGrid(n, plane_vertices, plane_indexes);
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

	diffuse_width, diffuse_height, diffuse_channel;
	stbi_uc* diffuse_pixels = stbi_load("res/textures/diffuse.jpg", &diffuse_width, &diffuse_height, &diffuse_channel, STBI_rgb_alpha);
	diffuse_size = diffuse_width * diffuse_height * 4;
	if (!diffuse_pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	normal_width, normal_height, normal_channel;
	stbi_uc* normal_pixels = stbi_load("res/textures/normal.jpg", &normal_width, &normal_height, &normal_channel, STBI_rgb_alpha);
	normal_size = normal_width * normal_height * 4;
	if (!normal_pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	stbi_uc* pixels = hdx::loadTexture("res/textures/brickwall.jpg", plane_image_width, plane_image_height, plane_image_channel, plane_image_size);
	uint64_t each_img_size = plane_image_width * plane_image_height * 4;
	plane_pixels.insert(plane_pixels.end(), pixels, pixels + each_img_size);
	stbi_image_free(pixels);
	pixels = hdx::loadTexture("res/textures/brickwall_normal.jpg", plane_image_width, plane_image_height, plane_image_channel, plane_image_size);
	plane_pixels.insert(plane_pixels.end(), pixels, pixels + each_img_size);
	stbi_image_free(pixels);

	sampler = hdx::createTextureSampler(device, device_desc.properties, 1);
	hdx::createImageDesc(device, diffuse_image_desc, vk::Format::eR8G8B8A8Srgb, diffuse_width, diffuse_height, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	hdx::createImageDesc(device, normal_image_desc, vk::Format::eR8G8B8A8Srgb, normal_width, normal_height, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	hdx::createImageDesc(device, plane_texture, vk::Format::eR8G8B8A8Srgb, plane_image_width, plane_image_width, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2DArray, 2, {}, device_desc, 1);

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

	plane_vb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, plane_vertex_count * sizeof(Vertex));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, plane_vb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, plane_vb, plane_vertices.data(), plane_vertex_count * sizeof(Vertex));

	plane_ib = hdx::createBuffer(device, vk::BufferUsageFlagBits::eIndexBuffer, sizeof(uint32_t) * plane_index_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, plane_ib, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, plane_ib, plane_indexes.data(), sizeof(uint32_t)* plane_index_count);

	plane_tb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eTransferSrc, plane_image_size * 2);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, plane_tb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, plane_tb, plane_pixels.data(), plane_image_size * 2);

	plane_transform_ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MVP));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, plane_transform_ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, plane_transform_ub, &mvp, sizeof(MVP));

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

		hdx::transitionImageLayout(device, plane_texture, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, s_cmd, 1, 2);
		hdx::copyBufferToImage(device, plane_tb, plane_texture, plane_image_width, plane_image_height, 2, s_cmd);
		hdx::transitionImageLayout(device, plane_texture, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, s_cmd, 1, 2);
	hdx::endSingleTimeCommands(device, s_cmd, command_pool, queue);


	_DBI_u = hdx::createDescriptorBufferInfo(ub, sizeof(MVP));
	_DII_diffuse = hdx::createDescriptorImageInfo(diffuse_image_desc, sampler, vk::ImageLayout::eShaderReadOnlyOptimal);
	_DII_normal = hdx::createDescriptorImageInfo(normal_image_desc, sampler, vk::ImageLayout::eShaderReadOnlyOptimal);
	_DBI_light = hdx::createDescriptorBufferInfo(light_ub, sizeof(Light));
	_DBI_plane = hdx::createDescriptorBufferInfo(plane_transform_ub, sizeof(MVP));
	_DII_plane = hdx::createDescriptorImageInfo(plane_texture, sampler, vk::ImageLayout::eShaderReadOnlyOptimal);
	pool_sizes = {
		vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 5),
		vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 4)
	};
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, 3);

	_DSLB = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(3, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment)
	};
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);
	_WDS = {
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI_u, 0),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, _DII_diffuse, 1),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, _DII_normal, 2),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI_light, 3)
	};
	device.updateDescriptorSets(4, _WDS.data(), 0, nullptr);

	_DSLB_plane = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(3, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
	};
	_DSL_plane = hdx::createDescriptorSetLayout(device, _DSLB_plane);
	_DS_plane = hdx::allocateDescriptorSet(device, _DSL_plane, descriptor_pool);
	_WDS = {
		hdx::createWriteDescriptorSet(_DS_plane, vk::DescriptorType::eUniformBuffer, _DBI_plane, 0),
		hdx::createWriteDescriptorSet(_DS_plane, vk::DescriptorType::eCombinedImageSampler, _DII_plane, 1),
		hdx::createWriteDescriptorSet(_DS_plane, vk::DescriptorType::eUniformBuffer, _DBI_light, 2),
		hdx::createWriteDescriptorSet(_DS_plane, vk::DescriptorType::eCombinedImageSampler, _DII_shadow, 3)
	};
	device.updateDescriptorSets(4, _WDS.data(), 0, nullptr);


	_DSLB_shadow = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex),
	};
	_DSL_shadow = hdx::createDescriptorSetLayout(device, _DSLB_shadow);
	_DS_shadow = hdx::allocateDescriptorSet(device, _DSL_shadow, descriptor_pool);
	_WDS = {
		hdx::createWriteDescriptorSet(_DS_shadow, vk::DescriptorType::eUniformBuffer, _DBI_light, 0)
	};
	device.updateDescriptorSets(1, _WDS.data(), 0, nullptr);


	pipeline = hdx::createGraphicsPipeline(device, pipeline_layout, renderpass, msaa_samples, "res/shaders/shader.vert.spv", "res/shaders/shader.frag.spv", binding_descriptions, attribute_descriptions, _DSL, vk::PrimitiveTopology::eTriangleList, extent);
	plane_pipeline = hdx::createGraphicsPipeline(device, plane_PL, renderpass, msaa_samples, "res/shaders/plane.vert.spv", "res/shaders/plane.frag.spv", binding_descriptions, attribute_descriptions, _DSL_plane, vk::PrimitiveTopology::eTriangleList, extent);
	shadow_pipeline = hdx::createGraphicsPipeline(device, shadow_PL, shadow_rp, vk::SampleCountFlagBits::e1, "res/shaders/shadow.vert.spv", "res/shaders/shadow.frag.spv", binding_descriptions, attribute_descriptions, _DSL_shadow, vk::PrimitiveTopology::eTriangleList, extent);

	image_available_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	hdx::scale(plane_mvp.model, 5, 5, 5);
	hdx::translate(mvp.model, 0, 2, 0);
//	hdx::rotate(plane_mvp.model, 180, 0, 0);
	camera3D.translate(0, 1, 0);
}


void Application::update(float delta_time, AppState& app_state)
{
//	hdx::rotate(mvp.model, 0, 2, 0);
	mvp.view = camera3D.getViewMatrix();
	plane_mvp.view = camera3D.getViewMatrix();
	hdx::copyToDevice(device, ub, &mvp, sizeof(MVP));
	hdx::copyToDevice(device, plane_transform_ub, &plane_mvp, sizeof(MVP));
	float rv = 0.02f * delta_time;
	float mv = 0.002f * delta_time;
	if (Input::GetKey(Input::KEY_I))
	{
		camera3D.rotate(+rv, 0, 0);
	}
	if (Input::GetKey(Input::KEY_J))
	{
		camera3D.rotate(0, rv, 0);
	}
	if (Input::GetKey(Input::KEY_K))
	{
		camera3D.rotate(-rv, 0, 0);
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

	vk::Buffer vertex_buffers[] = { vb.buffer }, plane_vertex_buffers[] = {plane_vb.buffer};
	uint64_t offsets[] = { 0 }, plane_offsets[] = { 0 };

	std::vector<vk::ClearValue> clear_values = {
		vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
		vk::ClearDepthStencilValue(1.0f, 0)
	};

	hdx::beginSingleTimeCommands(device, sh_cmd);
	hdx::transitionImageLayout(sh_cmd, shadow_map, vk::Format::eD32Sfloat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eEarlyFragmentTests);

	hdx::beginRenderpass(command_buffer, shadow_rp, shadow_fb, 1024, 1024, { vk::ClearDepthStencilValue(1.0f, 0) });
	hdx::recordCommandBuffer(shadow_pipeline, shadow_PL, index_count, command_buffer, vertex_buffers, ib.buffer, _DS_shadow, offsets, 1, 1);
	hdx::recordCommandBuffer(shadow_pipeline, shadow_PL, plane_index_count, command_buffer, plane_vertex_buffers, plane_ib.buffer, _DS_shadow, plane_offsets, 1, 1);
	hdx::endRenderpass(command_buffer);

	hdx::transitionImageLayout(sh_cmd, shadow_map, vk::Format::eD32Sfloat, vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageLayout::eDepthStencilReadOnlyOptimal, vk::PipelineStageFlagBits::eEarlyFragmentTests, vk::PipelineStageFlagBits::eFragmentShader);
	hdx::endSingleTimeCommands(device, sh_cmd, command_pool, queue);

	hdx::beginRenderpass(command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
		hdx::recordCommandBuffer(pipeline, pipeline_layout, index_count, command_buffer, vertex_buffers, ib.buffer, _DS, offsets, 1, 1);
		hdx::recordCommandBuffer(plane_pipeline, plane_PL, plane_index_count, command_buffer, plane_vertex_buffers, plane_ib.buffer, _DS_plane, plane_offsets, 1, 1);
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
	hdx::cleanupBuffer(device, plane_vb);
	hdx::cleanupBuffer(device, plane_transform_ub);
	hdx::cleanupBuffer(device, plane_tb);
	hdx::cleanupBuffer(device, plane_ib);
	hdx::cleanupBuffer(device, diffuse_tb);
	hdx::cleanupBuffer(device, normal_tb);
	hdx::cleanupImage(device, diffuse_image_desc);
	hdx::cleanupImage(device, plane_texture);
	hdx::cleanupImage(device, normal_image_desc);
	hdx::cleanupImage(device, shadow_map);
	device.destroySampler(sampler);
	device.destroySampler(ssampler);

	device.destroyPipeline(pipeline);
	device.destroyPipelineLayout(pipeline_layout);
	device.destroyPipeline(plane_pipeline);
	device.destroyPipelineLayout(plane_PL);
	device.destroyPipeline(shadow_pipeline);
	device.destroyPipelineLayout(shadow_PL);

	device.destroy(shadow_fb);
	device.destroyRenderPass(renderpass);
	device.destroyRenderPass(shadow_rp);

	device.destroyDescriptorPool(descriptor_pool);
	device.destroyDescriptorSetLayout(_DSL);
	device.destroyDescriptorSetLayout(_DSL_plane);
	device.destroyDescriptorSetLayout(_DSL_shadow);

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