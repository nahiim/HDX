
#include "Application.h"



void computeLightSpaceMatrix(
	const glm::mat4& cameraProj,      // Camera projection matrix
	const glm::mat4& cameraView,      // Camera view matrix
	const glm::vec3& lightDir,         // Normalized direction of the light
	glm::mat4& lightView, glm::mat4& lightProjection
) {
	// Step 1: Inverse camera matrix to get frustum corners in world space
	glm::mat4 invCam = glm::inverse(cameraProj * cameraView);

	std::vector<glm::vec3> frustumCorners;
	for (int x = 0; x < 2; ++x)
		for (int y = 0; y < 2; ++y)
			for (int z = 0; z < 2; ++z) {
				glm::vec4 cornerNDC = glm::vec4(
					2.0f * x - 1.0f,
					2.0f * y - 1.0f,
					2.0f * z - 1.0f,
					1.0f
				);
				glm::vec4 worldPos = invCam * cornerNDC;
				worldPos /= worldPos.w;
				frustumCorners.push_back(glm::vec3(worldPos));
			}

	// Step 2: Create temporary light view matrix
	glm::mat3 lightBasis = glm::mat3(glm::lookAt(glm::vec3(0.0f), lightDir, glm::vec3(0, 1, 0)));

	// Step 3: Transform frustum corners to light space
	std::vector<glm::vec3> frustumCornersLS;
	for (const auto& corner : frustumCorners) {
		frustumCornersLS.push_back(lightBasis * corner);  // 3x3 rotation only
	}

	// Step 4: Compute AABB in light space
	glm::vec3 min = frustumCornersLS[0];
	glm::vec3 max = frustumCornersLS[0];
	for (int i = 1; i < 8; ++i) {
		min = glm::min(min, frustumCornersLS[i]);
		max = glm::max(max, frustumCornersLS[i]);
	}

	// Step 5: Recompute light view using the center of the frustum
	glm::vec3 frustumCenter = (min + max) * 0.5f;
	glm::vec3 lightPos = frustumCenter - lightDir * 100.0f;

	lightView = glm::lookAt(lightPos, frustumCenter, glm::vec3(0, 1, 0));
	lightProjection = glm::ortho(min.x, max.x, min.y, max.y, -max.z - 100.0f, -min.z + 100.0f);
}

// Function to generate a plane grid with indices
void generatePlaneGrid(int gridSize, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, glm::vec3 color)
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

			// Set the normal and tangent
			vertex.normal = normal;

			vertex.color = glm::vec4(color, 1.0f);

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
	window = new Window("Shadow Mapping", WIDTH, HEIGHT);
	window->getExtensions();
	hdx::createInstance(instance, window, "Shadow", enable_validation_layers, validation_layers);
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
	camera3D = PerspectiveCamera(glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::radians(45.0f), aspectRatio, 0.1f, 1000.0f);
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

	hdx::scale(plane_mvp.model, 5, 5, 5);
	hdx::translate(mvp.model, 0, 2, 0);

	light.color = glm::vec4(1, 1, 1, 0);
	light.position = glm::vec4(5,5,5, 1.0f);

	glm::vec3 cam_pos = camera3D.getPosition();
	glm::vec3 cam_target = glm::vec3(0, 0, 0);
	glm::vec3 cam_front = glm::normalize(cam_target - cam_pos);

	// Light comes from camera direction
	glm::vec3 lightDir = -cam_front;
	glm::vec3 lightTarget = cam_pos + cam_front * 10.0f;        // focus where camera is looking
	glm::vec3 lightPos = lightTarget - lightDir * 17.0f;         // move light back along lightDir

	light.mvp.view = glm::lookAt(lightPos, lightTarget, glm::vec3(0.0f, 1.0f, 0.0f));
	float nearPlane = -10.0f, farPlane = 50.0f;
	float orthoSize = 5;
	light.mvp.projection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize,	nearPlane, farPlane);

	//light.mvp.projection = glm::ortho(-orthoSize, orthoSize, -orthoSize, orthoSize, 0.1f, 50.0f);
	//light.mvp.view = glm::lookAt(cam_pos, cam_target, glm::vec3(0, 1, 0));

	float sz = 10.0f;
	//PerspectiveCamera cam = PerspectiveCamera(glm::vec3(5.0f, 5.0f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::radians(45.0f), sw / sw, 0.1f, 1000.0f);
	//light.mvp.projection = cam.getProjectionMatrix();
	//light.mvp.view = cam.getViewMatrix();
	//light.mvp.projection[1][1] *= -1;

	//light.mvp.projection = glm::perspective(glm::radians(45.0f), sw/sw, 1.0f, 7.5f);
	//light.mvp.projection = ca.getProjectionMatrix();
	//light.mvp.view = ca.getViewMatrix();

	std::cout << "projection:\n" << glm::to_string(light.mvp.projection) << std::endl;
	std::cout << "view:\n" << glm::to_string(light.mvp.view) << std::endl;

	//light.mvp = { glm::mat4(1.0f), mvp.view, mvp.projection, light.position };
	//light.mvp.projection[1][1] = -1;

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
	createImageDesc(device, shadow_map, vk::Format::eD32Sfloat, sw, sw, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
	shadow_fb = hdx::createFramebuffer(device, shadow_map.imageview, shadow_rp, sw, sw);
	ssampler = hdx::createShadowSampler(device, device_desc.properties);
	_DII_shadow = hdx::createDescriptorImageInfo(shadow_map, ssampler, vk::ImageLayout::eDepthStencilReadOnlyOptimal);


	binding_descriptions = { hdx::getBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex) };
	attribute_descriptions = {
		hdx::getAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, position)),
		hdx::getAttributeDescription(0, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, normal)),
		hdx::getAttributeDescription(0, 2, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex, color))
	};

	generatePlaneGrid(n, plane_vertices, plane_indexes, glm::vec3(0.2f, 0.9f, 0.1f));
	//generatePlaneGrid(8, vertices, indices, glm::vec3(0.9f, 0.1f, 0.1f));
	vertices.resize(vertex_count);
	indices.resize(index_count);
	hdx::fillGrid(positions, normals, uv, indices, tangents, stacks, slices);
	for (uint32_t i = 0; i < vertex_count; i++)
	{
		vertices[i].position = glm::vec4(positions[i], 1.0f);
		vertices[i].normal = glm::vec4(normals[i], 1.0f);
		vertices[i].color = glm::vec4(0.9f, 0.1f, 0.1f, 1.0);
	}


	vb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, vertex_count * sizeof(Vertex));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, vb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, vb, vertices.data(), vertex_count * sizeof(Vertex));

	ib = hdx::createBuffer(device, vk::BufferUsageFlagBits::eIndexBuffer, sizeof(uint32_t) * index_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ib, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ib, indices.data(), sizeof(uint32_t) * index_count);

	ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MVP));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, ub, &mvp, sizeof(MVP));

	plane_vb = hdx::createBuffer(device, vk::BufferUsageFlagBits::eVertexBuffer, plane_vertex_count * sizeof(Vertex));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, plane_vb, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, plane_vb, plane_vertices.data(), plane_vertex_count * sizeof(Vertex));

	plane_ib = hdx::createBuffer(device, vk::BufferUsageFlagBits::eIndexBuffer, sizeof(uint32_t) * plane_index_count);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, plane_ib, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, plane_ib, plane_indexes.data(), sizeof(uint32_t) * plane_index_count);

	plane_transform_ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(MVP));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, plane_transform_ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, plane_transform_ub, &plane_mvp, sizeof(MVP));

	light_ub = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Light));
	hdx::allocateBufferMemory(device, device_desc.memory_properties, light_ub, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, light_ub, &light, sizeof(Light));

	_DBI_u = hdx::createDescriptorBufferInfo(ub, sizeof(MVP));
	_DBI_light = hdx::createDescriptorBufferInfo(light_ub, sizeof(Light));
	_DBI_plane = hdx::createDescriptorBufferInfo(plane_transform_ub, sizeof(MVP));


	int texWidth, texHeight, texChannels;
	stbi_uc* pixels = stbi_load("res/textures/tile.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	uint64_t texture_size = texWidth * texHeight * 4;
	if (!pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}

	bf = hdx::createBuffer(device, vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferSrc, texWidth * texHeight * 4);
	hdx::allocateBufferMemory(device, device_desc.memory_properties, bf, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
	hdx::copyToDevice(device, bf, pixels, texWidth * texHeight * 4);

	createImageDesc(device, im, vk::Format::eR8G8B8A8Srgb, sw, sw, vk::SampleCountFlagBits::e1, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);

	//hdx::beginSingleTimeCommands(device, command_buffer);
	//hdx::transitionImageLayout(device, im.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::Format::eR8G8B8A8Srgb, command_buffer, 1, 1);
	//hdx::copyBufferToImage(device, bf, im, texWidth, texHeight, 1, command_buffer);
	//hdx::transitionImageLayout(device, im.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::Format::eR8G8B8A8Srgb, command_buffer, 1, 1);
	//hdx::endSingleTimeCommands(device, command_buffer, command_pool, queue);

	sa = hdx::createTextureSampler(device, device_desc.properties, 1);
	ii = hdx::createDescriptorImageInfo(im, sa, vk::ImageLayout::eShaderReadOnlyOptimal);



	pool_sizes = {
		vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 9),
		vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 4)
	};
	descriptor_pool = hdx::createDescriptorPool(device, pool_sizes, 9);

	_DSLB = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
	};
	_DSL = hdx::createDescriptorSetLayout(device, _DSLB);
	_DS = hdx::allocateDescriptorSet(device, _DSL, descriptor_pool);
	_WDS = {
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI_u, 0),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eUniformBuffer, _DBI_light, 1),
		hdx::createWriteDescriptorSet(_DS, vk::DescriptorType::eCombinedImageSampler, _DII_shadow, 2)
	};
	device.updateDescriptorSets(3, _WDS.data(), 0, nullptr);



	_DSLB_plane = {
	hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
	hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment),
	hdx::createDescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
	};
	_DSL_plane = hdx::createDescriptorSetLayout(device, _DSLB_plane);
	_DS_plane = hdx::allocateDescriptorSet(device, _DSL_plane, descriptor_pool);
	_WDS = {
		hdx::createWriteDescriptorSet(_DS_plane, vk::DescriptorType::eUniformBuffer, _DBI_plane, 0),
		hdx::createWriteDescriptorSet(_DS_plane, vk::DescriptorType::eUniformBuffer, _DBI_light, 1),
		hdx::createWriteDescriptorSet(_DS_plane, vk::DescriptorType::eCombinedImageSampler, _DII_shadow, 2)
	};
	device.updateDescriptorSets(3, _WDS.data(), 0, nullptr);


	_DSLB_shadow = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex)
	};
	_DSL_shadow = hdx::createDescriptorSetLayout(device, _DSLB_shadow);
	_DS_shadow = hdx::allocateDescriptorSet(device, _DSL_shadow, descriptor_pool);
	_WDS = {
		hdx::createWriteDescriptorSet(_DS_shadow, vk::DescriptorType::eUniformBuffer, _DBI_light, 0),
		hdx::createWriteDescriptorSet(_DS_shadow, vk::DescriptorType::eUniformBuffer, _DBI_plane, 1)
	};
	device.updateDescriptorSets(2, _WDS.data(), 0, nullptr);

	_DSLB_s = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex)
	};
	_DSL_s = hdx::createDescriptorSetLayout(device, _DSLB_s);
	_DS_s = hdx::allocateDescriptorSet(device, _DSL_s, descriptor_pool);
	_WDS = {
		hdx::createWriteDescriptorSet(_DS_s, vk::DescriptorType::eUniformBuffer, _DBI_light, 0),
		hdx::createWriteDescriptorSet(_DS_s, vk::DescriptorType::eUniformBuffer, _DBI_u, 1)
	};
	device.updateDescriptorSets(2, _WDS.data(), 0, nullptr);


	_DSLB_quad = {
		hdx::createDescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment),
		hdx::createDescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eFragment)
	};
	_DSL_quad = hdx::createDescriptorSetLayout(device, _DSLB_quad);
	_DS_quad = hdx::allocateDescriptorSet(device, _DSL_quad, descriptor_pool);
	_WDS = {
		hdx::createWriteDescriptorSet(_DS_quad, vk::DescriptorType::eUniformBuffer, _DBI_light, 0),
		hdx::createWriteDescriptorSet(_DS_quad, vk::DescriptorType::eCombinedImageSampler, _DII_shadow, 1)
	};
	device.updateDescriptorSets(2, _WDS.data(), 0, nullptr);

	pipeline_layout = hdx::createPipelineLayout(device, _DSL, 0);
	plane_PL = hdx::createPipelineLayout(device, _DSL_plane, 0);
	shadow_PL = hdx::createPipelineLayout(device, _DSL_shadow, 0);
	s_PL = hdx::createPipelineLayout(device, _DSL_s, 0);
	quad_PL = hdx::createPipelineLayout(device, _DSL_quad, 0);
	pipeline = hdx::createGraphicsPipeline(device, pipeline_layout, renderpass, msaa_samples, "res/shaders/plane.vert.spv", "res/shaders/ball.frag.spv", binding_descriptions, attribute_descriptions, _DSL, vk::PrimitiveTopology::eTriangleList, extent);
	plane_pipeline = hdx::createGraphicsPipeline(device, plane_PL, renderpass, msaa_samples, "res/shaders/plane.vert.spv", "res/shaders/plane.frag.spv", binding_descriptions, attribute_descriptions, _DSL_plane, vk::PrimitiveTopology::eTriangleList, extent);
	quad_pipeline = hdx::createGraphicsPipeline(device, quad_PL, renderpass, msaa_samples, "res/shaders/quad.vert.spv", "res/shaders/quad.frag.spv", binding_descriptions, attribute_descriptions, _DSL_quad, vk::PrimitiveTopology::eTriangleList, extent);
	shadow_pipeline = hdx::createGraphicsPipeline(device, shadow_PL, shadow_rp, vk::SampleCountFlagBits::e1, "res/shaders/shadow.vert.spv", "res/shaders/shadow.frag.spv", binding_descriptions, attribute_descriptions, _DSL_shadow, vk::PrimitiveTopology::eTriangleList, extent);
	s_pipeline = hdx::createGraphicsPipeline(device, s_PL, shadow_rp, vk::SampleCountFlagBits::e1, "res/shaders/shadow.vert.spv", "res/shaders/shadow.frag.spv", binding_descriptions, attribute_descriptions, _DSL_s, vk::PrimitiveTopology::eTriangleList, extent);

	image_available_semaphore = hdx::createSemaphore(device);
	in_between_semaphore = hdx::createSemaphore(device);
	render_finished_semaphore = hdx::createSemaphore(device);
	in_flight_fence = hdx::createFence(device);
	dp_fence = hdx::createFence(device);

	wait_stages[0] = vk::PipelineStageFlagBits::eVertexInput;
	wait_stages[1] = vk::PipelineStageFlagBits::eColorAttachmentOutput;

	//	hdx::rotate(plane_mvp.model, 180, 0, 0);
	camera3D.translate(0, 1, 0);


	//hdx::beginSingleTimeCommands(device, command_buffer);
	//hdx::transitionImageLayout(command_buffer, shadow_map.image, vk::Format::eD32Sfloat, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilReadOnlyOptimal, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eFragmentShader);
	//hdx::endSingleTimeCommands(device, command_buffer, command_pool, queue);
}


void Application::update(float delta_time, AppState& app_state)
{
	//	hdx::rotate(mvp.model, 0, 2, 0);
	mvp.view = camera3D.getViewMatrix();
	plane_mvp.view = camera3D.getViewMatrix();
	mvp.view_pos = glm::vec4(camera3D.getPosition(), 1.0f);
	plane_mvp.view_pos = glm::vec4(camera3D.getPosition(), 1.0f);
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

	vk::Buffer vertex_buffers[] = { vb.buffer }, plane_vertex_buffers[] = { plane_vb.buffer };
	uint64_t offsets[] = { 0 }, plane_offsets[] = { 0 };

	std::vector<vk::ClearValue> clear_values = {
		vk::ClearColorValue(std::array<float, 4>{0.2f, 0.2f, 0.0f, 1.0f}),
		vk::ClearDepthStencilValue(1.0f, 0)
	};


	hdx::beginRenderpass(s_cmd, shadow_rp, shadow_fb, sw, sw, { vk::ClearDepthStencilValue(1.0f, 0) });
	hdx::recordCommandBuffer(s_pipeline, s_PL, index_count, s_cmd, vertex_buffers, ib.buffer, _DS_s, offsets, 1, 1);
	hdx::recordCommandBuffer(shadow_pipeline, shadow_PL, plane_index_count, s_cmd, plane_vertex_buffers, plane_ib.buffer, _DS_shadow, plane_offsets, 1, 1);
	hdx::endRenderpass(s_cmd);



	hdx::beginRenderpass(command_buffer, renderpass, framebuffers[image_index], extent, clear_values);
	hdx::recordCommandBuffer(pipeline, pipeline_layout, index_count, command_buffer, vertex_buffers, ib.buffer, _DS, offsets, 1, 1);
	hdx::recordCommandBuffer(plane_pipeline, plane_PL, plane_index_count, command_buffer, plane_vertex_buffers, plane_ib.buffer, _DS_plane, plane_offsets, 1, 1);
	//command_buffer.bindDescriptorSets(
	//	vk::PipelineBindPoint::eGraphics,
	//	quad_PL,
	//	0,
	//	_DS_quad,
	//	nullptr // dynamic offsets
	//);
	//command_buffer.bindPipeline(
	//	vk::PipelineBindPoint::eGraphics,
	//	quad_pipeline
	//);
	//command_buffer.draw(
	//	3,   // vertex count
	//	1,   // instance count
	//	0,   // first vertex
	//	0    // first instance
	//);
	hdx::endRenderpass(command_buffer);


	vk::SubmitInfo s_i{};
	s_i.sType = vk::StructureType::eSubmitInfo;
	s_i.commandBufferCount = 1;
	s_i.pCommandBuffers = &s_cmd;
	// Before submit:
	device.resetFences(dp_fence);
	queue.submit({ s_i }, dp_fence);
	// Wait for depth pass to finish
	device.waitForFences(dp_fence, VK_TRUE, UINT64_MAX);
	s_cmd.reset({ vk::CommandBufferResetFlagBits::eReleaseResources });

	vk::SubmitInfo submit_info{};
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

	frame_index++;
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
	hdx::cleanupBuffer(device, bf);
	hdx::cleanupImage(device, diffuse_image_desc);
	hdx::cleanupImage(device, plane_texture);
	hdx::cleanupImage(device, normal_image_desc);
	hdx::cleanupImage(device, shadow_map);
	device.destroySampler(sampler);
	hdx::cleanupImage(device, im);
	device.destroySampler(sa);
	device.destroySampler(ssampler);

	device.destroyPipeline(pipeline);
	device.destroyPipelineLayout(pipeline_layout);
	device.destroyPipeline(plane_pipeline);
	device.destroyPipelineLayout(plane_PL);
	device.destroyPipeline(shadow_pipeline);
	device.destroyPipelineLayout(shadow_PL);
	device.destroyPipeline(quad_pipeline);
	device.destroyPipelineLayout(quad_PL);
	device.destroyPipeline(s_pipeline);
	device.destroyPipelineLayout(s_PL);

	device.destroy(shadow_fb);
	device.destroyRenderPass(renderpass);
	device.destroyRenderPass(shadow_rp);

	device.destroyDescriptorPool(descriptor_pool);
	device.destroyDescriptorSetLayout(_DSL);
	device.destroyDescriptorSetLayout(_DSL_plane);
	device.destroyDescriptorSetLayout(_DSL_shadow);
	device.destroyDescriptorSetLayout(_DSL_quad);
	device.destroyDescriptorSetLayout(_DSL_s);

	device.destroySemaphore(render_finished_semaphore);
	device.destroySemaphore(in_between_semaphore);
	device.destroySemaphore(image_available_semaphore);
	device.destroyFence(in_flight_fence);
	device.destroyFence(dp_fence);

	device.destroyCommandPool(command_pool);

	device.destroy();
	instance.destroySurfaceKHR(surface);

	if (enable_validation_layers) { instance.destroyDebugUtilsMessengerEXT(debug_messenger, nullptr, dldi); }

	instance.destroy();	std::cout << "Application DESTROYED!!\n";

	delete window;
}