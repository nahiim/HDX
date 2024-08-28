
#include <hdx/hdx.hpp>

namespace hdx
{
	std::vector<char> read_file(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("failed to open file!");
		}

		size_t file_size = (size_t)file.tellg();
		std::vector<char> buffer(file_size);

		file.seekg(0);
		file.read(buffer.data(), file_size);

		file.close();



		return buffer;
	}


	





    vk::VertexInputBindingDescription getBindingDescription(uint32_t binding, uint32_t stride, vk::VertexInputRate input_rate)
    {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = binding;
        bindingDescription.stride = stride;
        bindingDescription.inputRate = input_rate;

        return bindingDescription;
    }
    vk::VertexInputAttributeDescription getAttributeDescription(uint32_t binding, uint32_t location, vk::Format format, uint32_t offset)
    {
        vk::VertexInputAttributeDescription attributeDescription;
        attributeDescription.binding = binding;
        attributeDescription.location = location;
        attributeDescription.format = format;
        attributeDescription.offset = offset;

        return attributeDescription;
    }
	
	VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
	{
		std::cerr << "validation ERRRRRR: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}
	void createDebugMessenger(vk::DebugUtilsMessengerEXT &debug_messenger, vk::Instance instance, vk::DispatchLoaderDynamic dldi)
	{
		vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
		createInfo.sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT;
		createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
		createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr; // Optional

		try
		{
			debug_messenger = instance.createDebugUtilsMessengerEXT(createInfo, nullptr, dldi);
		}
		catch (vk::SystemError err)
		{
			std::cerr << "Vulkan Error: " << err.what() << std::endl;
			std::cout << "Failed to setup Debug Messenger!\n";
		}
	}
	bool checkValidationLayerSupport(const std::vector<const char*> validation_layers)
	{
		uint32_t layer_count;
		if (vk::enumerateInstanceLayerProperties(&layer_count, nullptr) != vk::Result::eSuccess)
		{
			std::cout << "Failed to enumerate instance layer properties!\n";
		}

		std::vector<vk::LayerProperties> available_layers(layer_count);
		if (vk::enumerateInstanceLayerProperties(&layer_count, available_layers.data()) != vk::Result::eSuccess)
		{
			std::cout << "Failed to enumerate instance layer properties!\n";
		}

		std::cout << "\nSupported Layers:\n";

		for (const char* layer_name : validation_layers)
		{
		    bool layer_found = false;

		    for (const auto& layer_properties : available_layers)
		    {
				std::cout << "\t" << layer_properties.layerName << "\n";
		        if (strcmp(layer_name, layer_properties.layerName) == 0)
		        {
		            layer_found = true;
		            break;
		        }
		    }

		    if (!layer_found)
		    {
				std::cerr << "\nValidation layer not supported: " << layer_name << '\n';
		        return false;
		    }
		}

		std::cout << "\nValidation layer supported: " << validation_layers.data()[0] << '\n';

		return true;
	}
	void createInstance(vk::Instance &instance, Window *window, const char* app_name, bool enable_validation_layers, const std::vector<const char*> validation_layers)
	{	
		// Application info
		vk::ApplicationInfo app_info{};
		app_info.sType = vk::StructureType::eApplicationInfo;
		app_info.pApplicationName = app_name;
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pEngineName = "HDX Engine";
		app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.apiVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);


		// Instance Create Info
		vk::InstanceCreateInfo create_info {};
		create_info.sType = vk::StructureType::eInstanceCreateInfo;
		create_info.pApplicationInfo = &app_info;
		create_info.enabledExtensionCount = static_cast<uint32_t>(window->sdl_extensions.size());
		create_info.ppEnabledExtensionNames = window->sdl_extensions.data();
		create_info.enabledLayerCount = 0;



		uint32_t extension_count = 0;
		if (vk::enumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr) != vk::Result::eSuccess)
		{
			std::cout << "Failed to enumerate Extension layer properties!\n";
		}
		std::vector<vk::ExtensionProperties> extensions(extension_count);
		if (vk::enumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data()) != vk::Result::eSuccess)
		{
			std::cout << "Failed to enumerate Extension layer properties!\n";
		}

		std::cout << "available extensions:\n";
		for (const auto& extension : extensions)
		{
		    std::cout << '\t' << extension.extensionName << '\n';
		}
	    if (enable_validation_layers && !checkValidationLayerSupport(validation_layers))
	    {
	        throw std::runtime_error("validation layers requested, but not available!");
	    }
		if (enable_validation_layers)
		{
			std::cout << "\n" << validation_layers.data() << "\n";
		    create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
		    create_info.ppEnabledLayerNames = validation_layers.data();
		}
		else
		{
		    create_info.enabledLayerCount = 0;
		}
		// Now we can make the Vulkan instance
		try
		{
			std::cout << "\ncreating Instance...\n";
			instance = vk::createInstance(create_info, nullptr);
		}
		catch (std::exception err)
		{
			std::cout << "Failed to create Instance: " << err.what() <<  std::endl;
		}
	}

// MEMORY FUNCTIONS
    uint32_t findMemoryType(vk::PhysicalDeviceMemoryProperties& mem_properties, uint32_t type_filter, vk::MemoryPropertyFlags properties)
    {
        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++)
        {
            if (type_filter & (1 << i) && (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }
    void copyToDevice(vk::Device device, BufferDesc dst_buffer, void* source, uint64_t size)
    {
        void* data;
        device.mapMemory(dst_buffer.memory, 0, size, {}, &data);
        memcpy(data, source, (size_t)size);
        device.unmapMemory(dst_buffer.memory);
    }



// PIPELINE FUNCTIONS
	vk::ShaderModule createShaderModule(const vk::Device& device, const std::vector<char>& code)
	{
		vk::ShaderModuleCreateInfo create_info{};
		create_info.sType = vk::StructureType::eShaderModuleCreateInfo;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

		vk::ShaderModule shader_module;
		try
		{
			shader_module = device.createShaderModule(create_info);
			std::cout << "\n\n SHADER MODULE SUCCESSFULLY \n\n";
		}
		catch (vk::SystemError err)
		{
			throw std::runtime_error("failed to create Shader Module!");
		}



		return shader_module;
	}



	vk::Pipeline createGraphicsPipeline(
		const vk::Device& device, vk::PipelineLayout& pipeline_layout, const vk::RenderPass& rp, vk::SampleCountFlagBits msaa_samples,
		const std::string& vertex_shader, const std::string& fragment_shader,
		const std::vector<vk::VertexInputBindingDescription>& binding_descriptions, const std::vector<vk::VertexInputAttributeDescription>& attribute_descriptions,
		vk::DescriptorSetLayout dset_layout, vk::PrimitiveTopology topology, vk::Extent2D extent)
	{
		// Load shaders
		auto vert_shader_code = read_file(vertex_shader);
		auto frag_shader_code = read_file(fragment_shader);
		vk::ShaderModule vert_shader_module = createShaderModule(device, vert_shader_code);
		vk::ShaderModule frag_shader_module = createShaderModule(device, frag_shader_code);

		// Shader stage create info
		vk::PipelineShaderStageCreateInfo vert_shader_stage_info{};
		vert_shader_stage_info.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
		vert_shader_stage_info.stage = vk::ShaderStageFlagBits::eVertex;
		vert_shader_stage_info.module = vert_shader_module;
		vert_shader_stage_info.pName = "main";

		vk::PipelineShaderStageCreateInfo frag_shader_stage_info{};
		frag_shader_stage_info.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
		frag_shader_stage_info.stage = vk::ShaderStageFlagBits::eFragment;
		frag_shader_stage_info.module = frag_shader_module;
		frag_shader_stage_info.pName = "main";

		vk::PipelineShaderStageCreateInfo shader_stages[] = { vert_shader_stage_info, frag_shader_stage_info };

		// Vertex input state create info
		vk::PipelineVertexInputStateCreateInfo vertex_input_info{};
		vertex_input_info.sType = vk::StructureType::ePipelineVertexInputStateCreateInfo;
		vertex_input_info.vertexBindingDescriptionCount = static_cast<uint32_t>(binding_descriptions.size());
		vertex_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribute_descriptions.size());
		vertex_input_info.pVertexBindingDescriptions = binding_descriptions.data();
		vertex_input_info.pVertexAttributeDescriptions = attribute_descriptions.data();

		// Input assembly state create info
		vk::PipelineInputAssemblyStateCreateInfo input_assembly{};
		input_assembly.sType = vk::StructureType::ePipelineInputAssemblyStateCreateInfo;
		input_assembly.topology = topology;
		input_assembly.primitiveRestartEnable = VK_FALSE;

		vk::Viewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(extent.width);
		viewport.height = static_cast<float>(extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		vk::Rect2D scissor{};
		scissor.offset = vk::Offset2D{ 0, 0 };
		scissor.extent = extent;

		// Viewport and scissor state create info
		vk::PipelineViewportStateCreateInfo viewport_state{};
		viewport_state.sType = vk::StructureType::ePipelineViewportStateCreateInfo;
		viewport_state.viewportCount = 1;
		viewport_state.pViewports = &viewport;
		viewport_state.scissorCount = 1;
		viewport_state.pScissors = &scissor;

		// Rasterizer state create info
		vk::PipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = vk::StructureType::ePipelineRasterizationStateCreateInfo;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = vk::PolygonMode::eFill;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = vk::CullModeFlagBits::eNone;
		rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
		rasterizer.depthBiasEnable = VK_FALSE;

		// Multisampling state create info
		vk::PipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = vk::StructureType::ePipelineMultisampleStateCreateInfo;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = msaa_samples;

		// Color blend attachment state create info
		vk::PipelineColorBlendAttachmentState color_blend_attachment{};
		color_blend_attachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
		color_blend_attachment.blendEnable = VK_FALSE; // No blending

		vk::PipelineColorBlendStateCreateInfo color_blending{};
		color_blending.sType = vk::StructureType::ePipelineColorBlendStateCreateInfo;
		color_blending.logicOpEnable = VK_FALSE;
		color_blending.logicOp = vk::LogicOp::eCopy; // Optional
		color_blending.attachmentCount = 1;
		color_blending.pAttachments = &color_blend_attachment;
		color_blending.blendConstants[0] = 0.0f; // Optional
		color_blending.blendConstants[1] = 0.0f; // Optional
		color_blending.blendConstants[2] = 0.0f; // Optional
		color_blending.blendConstants[3] = 0.0f; // Optional

		// Dynamic state create info
		std::vector<vk::DynamicState> dynamic_states = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};

		vk::PipelineDynamicStateCreateInfo dynamic_state{};
		dynamic_state.sType = vk::StructureType::ePipelineDynamicStateCreateInfo;
		dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
		dynamic_state.pDynamicStates = dynamic_states.data();

		// Pipeline layout create info
		vk::PipelineLayoutCreateInfo pipeline_layout_info{};
		pipeline_layout_info.sType = vk::StructureType::ePipelineLayoutCreateInfo;
		pipeline_layout_info.setLayoutCount = 1;
		pipeline_layout_info.pSetLayouts = &dset_layout;

		// Create pipeline layout
		try {
			pipeline_layout = device.createPipelineLayout(pipeline_layout_info);
		}
		catch (vk::SystemError err) {
			throw std::runtime_error("failed to create pipeline layout!");
		}

		// Depth stencil state create info
		vk::PipelineDepthStencilStateCreateInfo depth_stencil_info{};
		depth_stencil_info.sType = vk::StructureType::ePipelineDepthStencilStateCreateInfo;
		depth_stencil_info.depthTestEnable = VK_TRUE;
		depth_stencil_info.depthWriteEnable = VK_TRUE;
		depth_stencil_info.depthCompareOp = vk::CompareOp::eLessOrEqual;
		depth_stencil_info.depthBoundsTestEnable = VK_FALSE;
		depth_stencil_info.stencilTestEnable = VK_FALSE;

		// Graphics pipeline create info
		vk::GraphicsPipelineCreateInfo pipeline_info{};
		pipeline_info.sType = vk::StructureType::eGraphicsPipelineCreateInfo;
		pipeline_info.stageCount = 2;
		pipeline_info.pStages = shader_stages;
		pipeline_info.pVertexInputState = &vertex_input_info;
		pipeline_info.pInputAssemblyState = &input_assembly;
		pipeline_info.pViewportState = &viewport_state;
		pipeline_info.pRasterizationState = &rasterizer;
		pipeline_info.pMultisampleState = &multisampling;
		pipeline_info.pDepthStencilState = &depth_stencil_info;
		pipeline_info.pColorBlendState = &color_blending;
		pipeline_info.pDynamicState = &dynamic_state;
		pipeline_info.layout = pipeline_layout;
		pipeline_info.renderPass = rp;
		pipeline_info.subpass = 0;
		pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

		// Create the graphics pipeline
		vk::Pipeline pipeline;
		try {
			pipeline = device.createGraphicsPipeline(nullptr, pipeline_info).value;
		}
		catch (vk::SystemError err) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		// Destroy shader modules
		device.destroyShaderModule(frag_shader_module);
		device.destroyShaderModule(vert_shader_module);

		return pipeline;
	}
	vk::Pipeline createComputePipeline(const vk::Device& device, vk::DescriptorSetLayout dset_layout, vk::PipelineLayout& pipeline_layout, const std::string& path)
	{
		vk::Pipeline pipeline;

		auto compute_shader_code = read_file(path);

		vk::ShaderModule compute_shader_module = createShaderModule(device, compute_shader_code);

		vk::PipelineShaderStageCreateInfo computeShaderStageInfo{};
		computeShaderStageInfo.sType = vk::StructureType::ePipelineShaderStageCreateInfo;
		computeShaderStageInfo.stage = vk::ShaderStageFlagBits::eCompute;
		computeShaderStageInfo.module = compute_shader_module;
		computeShaderStageInfo.pName = "main";

		vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = vk::StructureType::ePipelineLayoutCreateInfo;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &dset_layout;

		// Create Pipeline Layout
		try
		{
			pipeline_layout = device.createPipelineLayout(pipelineLayoutInfo);
		}
		catch (vk::SystemError err)
		{
			throw std::runtime_error("failed to create pipeline layout!");
		}


		vk::ComputePipelineCreateInfo pipeline_info{};
		pipeline_info.sType = vk::StructureType::eComputePipelineCreateInfo;
		pipeline_info.layout = pipeline_layout;
		pipeline_info.stage = computeShaderStageInfo;

		// Create The compute Pipeline
		try
		{
			pipeline = device.createComputePipeline(nullptr, pipeline_info).value;
		}
		catch (vk::SystemError err)
		{
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		device.destroyShaderModule(compute_shader_module, nullptr);

		return pipeline;
	}




// RENDERPASS FUNCTIONS
	vk::RenderPass createRenderpass(vk::Device device, vk::SampleCountFlagBits msaa_samples, vk::Format format)
	{
		vk::RenderPass render_pass;

		vk::AttachmentDescription color_attachment{};
		color_attachment.format = vk::Format::eR8G8B8A8Srgb;
		color_attachment.samples = msaa_samples;
		color_attachment.loadOp = vk::AttachmentLoadOp::eClear;
		color_attachment.storeOp = vk::AttachmentStoreOp::eStore;
		color_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		color_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		color_attachment.initialLayout = vk::ImageLayout::eUndefined;
		color_attachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::AttachmentReference color_attachment_ref{};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;


		vk::AttachmentDescription depth_attachment{};
		depth_attachment.format = vk::Format::eD32Sfloat;
		depth_attachment.samples = msaa_samples;
		depth_attachment.loadOp = vk::AttachmentLoadOp::eClear;
		depth_attachment.storeOp = vk::AttachmentStoreOp::eDontCare;
		depth_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		depth_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		depth_attachment.initialLayout = vk::ImageLayout::eUndefined;
		depth_attachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentReference depth_attachment_ref{};
		depth_attachment_ref.attachment = 1;
		depth_attachment_ref.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::AttachmentDescription colorAttachmentResolve{};
		colorAttachmentResolve.format = format;
		colorAttachmentResolve.samples = vk::SampleCountFlagBits::e1;
		colorAttachmentResolve.loadOp = vk::AttachmentLoadOp::eDontCare;
		colorAttachmentResolve.storeOp = vk::AttachmentStoreOp::eStore;
		colorAttachmentResolve.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		colorAttachmentResolve.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		colorAttachmentResolve.initialLayout = vk::ImageLayout::eUndefined;
		colorAttachmentResolve.finalLayout = vk::ImageLayout::ePresentSrcKHR;

		vk::AttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

		vk::SubpassDescription subpass{};
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;
		subpass.pDepthStencilAttachment = &depth_attachment_ref;
		subpass.pResolveAttachments = &colorAttachmentResolveRef;

		vk::SubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
		dependency.srcAccessMask = {};
		dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;;
		dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

		//array of 2 attachments, one for the color, and other for depth
		vk::AttachmentDescription attachments[3] = { color_attachment, depth_attachment, colorAttachmentResolve };
//		vk::SubpassDependency dependencies[2] = { color_dependency, depth_dependency };


		vk::RenderPassCreateInfo renderpass_info{};
		renderpass_info.sType = vk::StructureType::eRenderPassCreateInfo;
		renderpass_info.attachmentCount = 3;
		renderpass_info.pAttachments = &attachments[0];
		renderpass_info.subpassCount = 1;
		renderpass_info.pSubpasses = &subpass;
		renderpass_info.dependencyCount = 1;
		renderpass_info.pDependencies = &dependency;


		try
		{
			render_pass = device.createRenderPass(renderpass_info);
		}
		catch (vk::SystemError err)
		{
			throw std::runtime_error("failed to create render pass!");
		}

		return render_pass;
	}

	vk::RenderPass createDepthRenderpass(vk::Device device)
	{
		vk::RenderPass render_pass;

		vk::AttachmentDescription depth_attachment{};
		depth_attachment.format = vk::Format::eD32Sfloat;
		depth_attachment.samples = vk::SampleCountFlagBits::e1;
		depth_attachment.loadOp = vk::AttachmentLoadOp::eClear;
		depth_attachment.storeOp = vk::AttachmentStoreOp::eStore;
		depth_attachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		depth_attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		depth_attachment.initialLayout = vk::ImageLayout::eUndefined;
		depth_attachment.finalLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;

		vk::AttachmentReference depth_attachment_ref{};
		depth_attachment_ref.attachment = 0;
		depth_attachment_ref.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

		vk::SubpassDescription subpass{};
		subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
		subpass.pDepthStencilAttachment = &depth_attachment_ref;

		vk::RenderPassCreateInfo renderpass_info{};
		renderpass_info.sType = vk::StructureType::eRenderPassCreateInfo;
		renderpass_info.attachmentCount = 1;
		renderpass_info.pAttachments = &depth_attachment;
		renderpass_info.subpassCount = 1;
		renderpass_info.pSubpasses = &subpass;


		try
		{
			render_pass = device.createRenderPass(renderpass_info);
		}
		catch (vk::SystemError err)
		{
			throw std::runtime_error("failed to create render pass!");
		}

		return render_pass;
	}




// SWAPCHAIN FUNCTIONS
    vk::SwapchainKHR createSwapchain(vk::Device device, vk::SurfaceKHR surface, vk::SurfaceFormatKHR format, vk::PresentModeKHR presentMode, vk::SurfaceCapabilitiesKHR capabilities, uint32_t width, uint32_t height, vk::Extent2D& ext)
    {
        // Determine the number of images in the swap chain
        uint32_t imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
        {
            imageCount = capabilities.maxImageCount;
        }


        vk::Extent2D extent;
        if (capabilities.currentExtent.width != UINT32_MAX)
        {
            extent = capabilities.currentExtent;
        }
        else
        {
            extent.width =
                std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, extent.width));
            extent.height =
                std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, extent.height));
        }

        ext = extent;



        // Create the swap chain
        vk::SwapchainCreateInfoKHR createInfo(
            {},
            surface,
            imageCount,
            format.format,
            format.colorSpace,
            extent,
            1,  // Number of layers
            vk::ImageUsageFlagBits::eColorAttachment,
            vk::SharingMode::eExclusive,  // Sharing mode
            0, nullptr,                    // Queue family indices (ignored when exclusive)
            capabilities.currentTransform,
            vk::CompositeAlphaFlagBitsKHR::eOpaque,
            presentMode,
            VK_TRUE,  // clipped
            nullptr   // old swap chain (nullptr for the first time)
        );

        return device.createSwapchainKHR(createInfo);
    }


    void cleanupSwapchain(vk::Device device, vk::SwapchainKHR swapchain, std::vector<vk::ImageView>& ivs, std::vector<vk::Framebuffer>& fbs,
        ImageDesc depth_image, ImageDesc color_image)
    {
        device.destroy(depth_image.image);
        device.freeMemory(depth_image.memory);
        device.destroy(depth_image.imageview);

        device.destroy(color_image.image);
        device.freeMemory(color_image.memory);
        device.destroy(color_image.imageview);

        for (size_t i = 0; i < fbs.size(); i++)
        {
            device.destroy(fbs[i]);
            device.destroy(ivs[i]);
        }

        device.destroy(swapchain);
    }


    void recreateSwapChain(vk::Device device, vk::SurfaceKHR surface, vk::SurfaceFormatKHR surface_format, vk::PresentModeKHR presentMode, vk::SurfaceCapabilitiesKHR capabilities, uint32_t width, uint32_t height, vk::Extent2D& ext, vk::SwapchainKHR& swapchain, ImageDesc& color_image, ImageDesc& depth_image, std::vector<vk::Image>& images, std::vector<vk::ImageView>& ivs, std::vector<vk::Framebuffer>& fbs, vk::Format format, vk::ImageAspectFlagBits aspect, vk::RenderPass rp, DeviceDesc& device_desc)
    {
        device.waitIdle();
        cleanupSwapchain(device, swapchain, ivs, fbs, depth_image, color_image);

        swapchain = createSwapchain(device, surface, surface_format, presentMode, capabilities, width, height, ext);

        createImageDesc(device, color_image, format, width, height, vk::SampleCountFlagBits::e8, vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);
        createImageDesc(device, depth_image, vk::Format::eD32Sfloat, width, height, vk::SampleCountFlagBits::e8, vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::ImageAspectFlagBits::eDepth, vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}, device_desc, 1);

        images = device.getSwapchainImagesKHR(swapchain);

        for (size_t i = 0; i < images.size(); i++)
        {
            ivs[i] = createImageView(device, images[i], format, aspect, 1, 1, vk::ImageViewType::e2D);
            fbs[i] = createFramebuffer(device, ivs[i], color_image.imageview, depth_image.imageview, rp, ext);
        }
    }




// SYNC OBJECTS
	vk::Semaphore createSemaphore(vk::Device device)
	{
		vk::Semaphore sem;

		vk::SemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = vk::StructureType::eSemaphoreCreateInfo;

		sem = device.createSemaphore(semaphoreInfo);

		return sem;
	}
	vk::Fence createFence(vk::Device device)
	{
		vk::Fence fence;

		vk::FenceCreateInfo fence_info{};
		fence_info.sType = vk::StructureType::eFenceCreateInfo;
		fence_info.flags = vk::FenceCreateFlagBits::eSignaled;

		fence = device.createFence(fence_info);

		return fence;
	}




// BUFFER FUNCTIONS
	BufferDesc createBuffer(vk::Device device, vk::BufferUsageFlags usage, uint64_t size)
	{
		BufferDesc buffer_desc;

		vk::BufferCreateInfo bufferInfo{};
		bufferInfo.sType = vk::StructureType::eBufferCreateInfo;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = vk::SharingMode::eExclusive;
		vk::Buffer buffer = device.createBuffer(bufferInfo, nullptr);

		buffer_desc.buffer = buffer;
		buffer_desc.size = size;
		return buffer_desc;
	}
	void allocateBufferMemory(const vk::Device& device, vk::PhysicalDeviceMemoryProperties mem_properties, BufferDesc& buffer_desc, vk::MemoryPropertyFlags properties)
	{
		// Get memory requirements
		vk::MemoryRequirements mem_requirements;
		device.getBufferMemoryRequirements(buffer_desc.buffer, &mem_requirements);

		vk::MemoryAllocateInfo allocInfo{};
		allocInfo.sType = vk::StructureType::eMemoryAllocateInfo;
		allocInfo.allocationSize = mem_requirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(mem_properties, mem_requirements.memoryTypeBits, properties);

		if (device.allocateMemory(&allocInfo, nullptr, &buffer_desc.memory) != vk::Result::eSuccess)
		{
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		device.bindBufferMemory(buffer_desc.buffer, buffer_desc.memory, 0);
	}
	void copyBuffer(vk::Device device, vk::CommandBuffer command_buffer, BufferDesc src_buffer, BufferDesc dst_buffer, uint64_t size)
	{
		vk::BufferCopy copyRegion{};
		copyRegion.size = size;
		command_buffer.copyBuffer(src_buffer.buffer, dst_buffer.buffer, 1, &copyRegion);
	}
	void cleanupBuffer(vk::Device& device, BufferDesc buffer_desc)
	{
		device.destroyBuffer(buffer_desc.buffer);
		device.freeMemory(buffer_desc.memory);
	}



// COMMAND FUNCTIONS
	vk::CommandPool createCommandPool(const vk::Device &device, const uint32_t &queueFamilyIndex)
	{
		vk::CommandPool command_pool;

		vk::CommandPoolCreateInfo pool_info{};
		pool_info.sType = vk::StructureType::eCommandPoolCreateInfo;
		pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = queueFamilyIndex;


		try
		{
			command_pool = device.createCommandPool(pool_info);
		}
		catch(vk::SystemError err)
		{
			throw std::runtime_error("failed to create command pool!");
		}

		return command_pool;
	}
	vk::CommandBuffer allocateCommandBuffer(const vk::Device& device, const vk::CommandPool command_pool)
	{
		vk::CommandBuffer command_buffer;

		vk::CommandBufferAllocateInfo alloc_info{};
		alloc_info.sType = vk::StructureType::eCommandBufferAllocateInfo;
		alloc_info.commandPool = command_pool;
		alloc_info.level = vk::CommandBufferLevel::ePrimary;
		alloc_info.commandBufferCount = 1;

		try
		{
			command_buffer = device.allocateCommandBuffers(alloc_info)[0];
		}
		catch(vk::SystemError err)
		{
			throw std::runtime_error("failed to create command buffer!");
		}

		return command_buffer;
	}
	void beginRenderpass(vk::CommandBuffer cmd_buffer, vk::RenderPass& renderpass, vk::Framebuffer framebuffer, vk::Extent2D extent, std::vector<vk::ClearValue> clear_values)
	{
		// Allocate command buffer recording information
		vk::CommandBufferBeginInfo beginInfo{};
		beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse; // Equivalent to VK_SUBPASS_CONTENTS_INLINE in the old Vulkan API.

		// Begin recording commands into the command buffer
		cmd_buffer.begin(beginInfo);

		// Begin render pass
		vk::RenderPassBeginInfo renderPassInfo{};
		renderPassInfo.renderPass = renderpass;
		renderPassInfo.framebuffer = framebuffer;
		renderPassInfo.renderArea.offset.x = 0;
		renderPassInfo.renderArea.offset.y = 0;
		renderPassInfo.renderArea.extent = extent;
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clear_values.size());
		renderPassInfo.pClearValues = clear_values.data();

		vk::Viewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)extent.width;
		viewport.height = (float)extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		cmd_buffer.setViewport(0, 1, &viewport);
		vk::Rect2D scissor{};
		scissor.offset = vk::Offset2D{ 0, 0 };
		scissor.extent = extent;
		cmd_buffer.setScissor(0, 1, &scissor);

		cmd_buffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
	}
	void beginRenderpass(vk::CommandBuffer cmd_buffer, vk::RenderPass& renderpass, vk::Framebuffer framebuffer, uint32_t width, uint32_t height, std::vector<vk::ClearValue> clear_values)
	{
		// Allocate command buffer recording information
		vk::CommandBufferBeginInfo beginInfo{};
		beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse; // Equivalent to VK_SUBPASS_CONTENTS_INLINE in the old Vulkan API.

		// Begin recording commands into the command buffer
		cmd_buffer.begin(beginInfo);

		// Begin render pass
		vk::RenderPassBeginInfo renderPassInfo{};
		renderPassInfo.renderPass = renderpass;
		renderPassInfo.framebuffer = framebuffer;
		renderPassInfo.renderArea.offset.x = 0;
		renderPassInfo.renderArea.offset.y = 0;
		renderPassInfo.renderArea.extent = vk::Extent2D(width, height);
		renderPassInfo.clearValueCount = static_cast<uint32_t>(clear_values.size());
		renderPassInfo.pClearValues = clear_values.data();

		vk::Viewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)width;
		viewport.height = (float)height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		cmd_buffer.setViewport(0, 1, &viewport);
		vk::Rect2D scissor{};
		scissor.offset = vk::Offset2D{ 0, 0 };
		scissor.extent = vk::Extent2D(width, height);
		cmd_buffer.setScissor(0, 1, &scissor);

		cmd_buffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
	}



	void recordCommandBuffer(vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout, uint32_t vertex_count, vk::CommandBuffer cmd_buffer, vk::Buffer vertex_buffers[], vk::DescriptorSet descriptor_set, uint64_t offsets[])
	{
		cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

		cmd_buffer.bindVertexBuffers(0, 1, vertex_buffers, offsets);
		cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
		cmd_buffer.draw(vertex_count, 1, 0, 0);
	}
	void recordCommandBuffer(vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout, uint32_t index_count, vk::CommandBuffer cmd_buffer, vk::Buffer vertex_buffers[], vk::Buffer index_buffer, vk::DescriptorSet descriptor_set, uint64_t offsets[], uint32_t binding_count, uint32_t instance_count)
	{
		cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

		cmd_buffer.bindVertexBuffers(0, binding_count, vertex_buffers, offsets);
		cmd_buffer.bindIndexBuffer(index_buffer, 0, vk::IndexType::eUint32);
		cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
		cmd_buffer.drawIndexed(index_count, instance_count, 0, 0, 0);
	}
	void recordCommandBuffer(vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout, uint32_t vertex_count, vk::CommandBuffer cmd_buffer, BufferDesc vertex_buffer_desc)
	{
		cmd_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

		vk::Buffer vertex_buffers[] = { vertex_buffer_desc.buffer };
		vk::DeviceSize offsets[] = { 0 };
		cmd_buffer.bindVertexBuffers(0, 1, vertex_buffers, offsets);
		cmd_buffer.draw(vertex_count, 1, 0, 0);
	}
	void recordComputeCommandBuffer(vk::Device device, vk::CommandBuffer cmd_buffer, vk::Pipeline pipeline, vk::PipelineLayout pipeline_layout, vk::DescriptorSet dsc_set, uint32_t x, uint32_t y, uint32_t z)
	{
		vk::CommandBufferBeginInfo beginInfo{};
		beginInfo.sType = vk::StructureType::eCommandBufferBeginInfo;

		cmd_buffer.begin(beginInfo);
		cmd_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
		cmd_buffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, 1, &dsc_set, 0, nullptr);
		cmd_buffer.dispatch(x, y, z);
		cmd_buffer.end();
	}
	void endRenderpass(vk::CommandBuffer command_buffer)
	{
		// End render pass
		command_buffer.endRenderPass();
		// End recording commands into the command buffer
		command_buffer.end();
	}
	void beginSingleTimeCommands(vk::Device device, vk::CommandBuffer& command_buffer)
	{
		vk::CommandBufferBeginInfo beginInfo{};
		beginInfo.sType = vk::StructureType::eCommandBufferBeginInfo;
		beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

		command_buffer.begin(beginInfo);
	}
	void endSingleTimeCommands(vk::Device device, vk::CommandBuffer &command_buffer, vk::CommandPool cmd_pool, vk::Queue queue)
	{
		command_buffer.end();

		vk::SubmitInfo submitInfo{};
		submitInfo.sType = vk::StructureType::eSubmitInfo;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &command_buffer;

		queue.submit({ submitInfo });
		queue.waitIdle();

//		device.freeCommandBuffers(cmd_pool, { command_buffer });
	}

	void submitCommand(vk::CommandBuffer command_buffer, vk::Queue queue, vk::Fence fence)
	{
		vk::SubmitInfo submit_info{};
		submit_info.sType = vk::StructureType::eSubmitInfo;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;
		queue.submit(submit_info, fence);
	}





// IMAGE FUNCTIONS
    vk::Image createImage(vk::Device device, uint32_t width, uint32_t height, uint32_t mip_levels, vk::SampleCountFlagBits num_samples, vk::Format format, vk::ImageUsageFlags usage, vk::ImageType image_type, uint32_t array_layers, vk::ImageCreateFlags flags)
    {
        vk::Image image;

        // CREATE IMAGE
        vk::ImageCreateInfo image_info;
        image_info.sType = vk::StructureType::eImageCreateInfo;
        image_info.pNext = nullptr;
        image_info.imageType = image_type;
        image_info.format = format;
        image_info.extent = vk::Extent3D{ width, height, 1 };
        image_info.mipLevels = mip_levels;
        image_info.arrayLayers = array_layers;
        image_info.samples = num_samples;
        image_info.tiling = vk::ImageTiling::eOptimal;
        image_info.usage = usage;
        image_info.flags = flags;
        image_info.initialLayout = vk::ImageLayout::eUndefined;

        image = device.createImage(image_info, nullptr);

        return image;
    }
	vk::ImageView createImageView(const vk::Device& device, vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspect, uint32_t mip_levels, uint32_t layer_count, vk::ImageViewType view_type, uint32_t base)
	{
		vk::ImageView imageView;

		// Specify image view creation info
		vk::ImageViewCreateInfo createInfo;
		createInfo.sType = vk::StructureType::eImageViewCreateInfo;
		createInfo.image = image;
		createInfo.viewType = view_type;
		createInfo.format = format;
		createInfo.components.r = vk::ComponentSwizzle::eIdentity;
		createInfo.components.g = vk::ComponentSwizzle::eIdentity;
		createInfo.components.b = vk::ComponentSwizzle::eIdentity;
		createInfo.components.a = vk::ComponentSwizzle::eIdentity;
		createInfo.subresourceRange.aspectMask = aspect;
		createInfo.subresourceRange.baseMipLevel = base;
		createInfo.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = layer_count;

		// Create image view
		try
		{
			imageView = device.createImageView(createInfo);
		}
		catch (vk::SystemError err)
		{
			std::cerr << "failed to create image view!" << err.what();
		}

		return imageView;
	}
    vk::ImageView createImageView(const vk::Device& device, vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspect, uint32_t mip_levels, uint32_t layer_count,vk::ImageViewType view_type)
    {
        vk::ImageView imageView;

        // Specify image view creation info
        vk::ImageViewCreateInfo createInfo;
        createInfo.sType = vk::StructureType::eImageViewCreateInfo;
        createInfo.image = image;
        createInfo.viewType = view_type;
        createInfo.format = format;
        createInfo.components.r = vk::ComponentSwizzle::eIdentity;
        createInfo.components.g = vk::ComponentSwizzle::eIdentity;
        createInfo.components.b = vk::ComponentSwizzle::eIdentity;
        createInfo.components.a = vk::ComponentSwizzle::eIdentity;
        createInfo.subresourceRange.aspectMask = aspect;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = layer_count;

        // Create image view
        try
        {
            imageView = device.createImageView(createInfo);
        }
        catch (vk::SystemError err)
        {
            std::cerr << "failed to create image view!" << err.what();
        }

        return imageView;
    }
    void allocateImageMemory(const vk::Device& device, vk::PhysicalDeviceMemoryProperties mem_properties, const vk::Image& image, vk::DeviceMemory& image_memory, vk::MemoryPropertyFlags properties)
    {
        // Get memory requirements
        vk::MemoryRequirements mem_requirements;
        device.getImageMemoryRequirements(image, &mem_requirements);

        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.sType = vk::StructureType::eMemoryAllocateInfo;
        allocInfo.allocationSize = mem_requirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(mem_properties, mem_requirements.memoryTypeBits, properties);

        if (device.allocateMemory(&allocInfo, nullptr, &image_memory) != vk::Result::eSuccess)
        {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        device.bindImageMemory(image, image_memory, 0);
    }
    void createImageDesc(vk::Device device, ImageDesc& image, vk::Format format, uint32_t width, uint32_t height, vk::SampleCountFlagBits msaa_samples, vk::ImageUsageFlags usage, vk::ImageAspectFlagBits aspect, vk::ImageType image_type, vk::ImageViewType view_type, uint32_t array_layers, vk::ImageCreateFlags flags, DeviceDesc device_desc, uint32_t mip_levels)
    {
        image.image = createImage(device, width, height, mip_levels, msaa_samples, format, usage, image_type, array_layers, flags);
        allocateImageMemory(device, device_desc.memory_properties, image.image, image.memory, vk::MemoryPropertyFlagBits::eDeviceLocal);
        image.imageview = createImageView(device, image.image, format, aspect, mip_levels, array_layers, view_type);
    }
    void cleanupImage(vk::Device device, ImageDesc image_desc)
    {
        device.destroy(image_desc.image);
        device.freeMemory(image_desc.memory);
        device.destroy(image_desc.imageview);
    }
    vk::Framebuffer createFramebuffer(const vk::Device& device, vk::ImageView swapchain_imageview, vk::ImageView color_imageview, const vk::ImageView& depth_imageview, const vk::RenderPass& rp, vk::Extent2D extent)
    {
        vk::Framebuffer framebuffer;

        vk::ImageView attachments[3];
        attachments[0] = color_imageview;
        attachments[1] = depth_imageview;
        attachments[2] = swapchain_imageview;

        vk::FramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType = vk::StructureType::eFramebufferCreateInfo;
        framebuffer_info.renderPass = rp;
        framebuffer_info.attachmentCount = 3;
        framebuffer_info.pAttachments = attachments;
        framebuffer_info.width = extent.width;
        framebuffer_info.height = extent.height;
        framebuffer_info.layers = 1;


        try
        {
            framebuffer = device.createFramebuffer(framebuffer_info);
        }
        catch (vk::SystemError err)
        {
            std::cerr << "failed to create framebuffer!" << err.what();
        }

        return framebuffer;
    }
	vk::Framebuffer createFramebuffer(const vk::Device& device, vk::ImageView imageview, const vk::RenderPass& rp, uint32_t width, uint32_t height)
	{
		vk::Framebuffer framebuffer;

		vk::FramebufferCreateInfo framebuffer_info{};
		framebuffer_info.sType = vk::StructureType::eFramebufferCreateInfo;
		framebuffer_info.renderPass = rp;
		framebuffer_info.attachmentCount = 1;
		framebuffer_info.pAttachments = &imageview;
		framebuffer_info.width = width;
		framebuffer_info.height = height;
		framebuffer_info.layers = 1;


		try
		{
			framebuffer = device.createFramebuffer(framebuffer_info);
		}
		catch (vk::SystemError err)
		{
			std::cerr << "failed to create framebuffer!" << err.what();
		}

		return framebuffer;
	}
	void transitionImageLayout(
		vk::CommandBuffer commandBuffer,
		ImageDesc texture,
		vk::Format format,
		vk::ImageLayout oldLayout,
		vk::ImageLayout newLayout,
		vk::PipelineStageFlags srcStage,
		vk::PipelineStageFlags dstStage)
	{

		vk::ImageMemoryBarrier barrier = {};
		barrier.sType = vk::StructureType::eImageMemoryBarrier;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = texture.image;
		barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		barrier.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
		barrier.dstAccessMask = vk::AccessFlagBits::eNoneKHR;

		if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
			barrier.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
			barrier.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
		}
		else if (oldLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
			barrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
		}

		commandBuffer.pipelineBarrier(
			srcStage,
			dstStage,
			vk::DependencyFlags(),
			0, nullptr,
			0, nullptr,
			1, &barrier
		);
	}
    void transitionImageLayout(vk::Device device, ImageDesc image_desc, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::Format format, vk::CommandBuffer command_buffer, uint32_t mip_levels, uint32_t layer_count)
    {
        vk::ImageMemoryBarrier barrier{};
        barrier.sType = vk::StructureType::eImageMemoryBarrier;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image_desc.image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = mip_levels;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = layer_count;

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlags();
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
		else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eTransfer;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		}
		else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlags();
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		}
		else if (oldLayout == vk::ImageLayout::eGeneral && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
		{
			barrier.srcAccessMask = vk::AccessFlags();
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
		}
		else if (oldLayout == vk::ImageLayout::eShaderReadOnlyOptimal && newLayout == vk::ImageLayout::eGeneral)
		{
			barrier.srcAccessMask = vk::AccessFlags();
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

			sourceStage = vk::PipelineStageFlagBits::eFragmentShader;
			destinationStage = vk::PipelineStageFlagBits::eComputeShader;
		}
		else if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eGeneral)
		{
			barrier.srcAccessMask = vk::AccessFlags();
			barrier.dstAccessMask = vk::AccessFlagBits::eShaderWrite;

			sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
			destinationStage = vk::PipelineStageFlagBits::eComputeShader;
		}

        command_buffer.pipelineBarrier(
            sourceStage, destinationStage,
            vk::DependencyFlags(),
            0, nullptr,
            0, nullptr,
            1, &barrier);
    }
    void copyBufferToImage(vk::Device device, BufferDesc buffer_desc, ImageDesc image_desc, uint32_t width, uint32_t height, uint32_t layer_count, vk::CommandBuffer command_buffer)
    {
        vk::BufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = layer_count;

        region.imageOffset = vk::Offset3D{ 0, 0, 0 };
        region.imageExtent = vk::Extent3D{ width, height, 1 };

        command_buffer.copyBufferToImage(
            buffer_desc.buffer,
            image_desc.image,
            vk::ImageLayout::eTransferDstOptimal,
            1,
            &region
        );
    }
    void generateMipmaps(vk::Device device, vk::CommandBuffer command_buffer, vk::Image image, vk::Format imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels, vk::FormatProperties format_properties)
    {
        if (!(format_properties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImageFilterLinear))
            throw std::runtime_error("texture image format does not support linear blitting!");
        
        vk::ImageMemoryBarrier barrier{};
        barrier.sType = vk::StructureType::eImageMemoryBarrier;
        barrier.image = image;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;

        int32_t mipWidth = texWidth;
        int32_t mipHeight = texHeight;

        for (uint32_t i = 1; i < mipLevels; i++)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

            command_buffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                vk::DependencyFlags(),
                0, nullptr,
                0, nullptr,
                1, &barrier);

            vk::ImageBlit blit{};
            blit.srcOffsets[0] = vk::Offset3D{ 0, 0, 0 };
            blit.srcOffsets[1] = vk::Offset3D{ mipWidth, mipHeight, 1 };
            blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = vk::Offset3D{ 0, 0, 0 };
            blit.dstOffsets[1] = vk::Offset3D{ mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
            blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            command_buffer.blitImage(
                image, vk::ImageLayout::eTransferSrcOptimal,
                image, vk::ImageLayout::eTransferDstOptimal,
                1, &blit,
                vk::Filter::eLinear);

            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            command_buffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
                vk::DependencyFlags(),
                0, nullptr,
                0, nullptr,
                1, &barrier);

            if (mipWidth > 1) mipWidth /= 2;
            if (mipHeight > 1) mipHeight /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

        command_buffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
            vk::DependencyFlags(),
            0, nullptr,
            0, nullptr,
            1, &barrier);
    }
    vk::ImageView createTextureImageView(vk::Device device, vk::Image image, vk::Format format, vk::ImageAspectFlagBits aspect, uint32_t mip_levels)
    {
        vk::ImageView iv = createImageView(device, image, format, aspect, mip_levels, 1, vk::ImageViewType::e2D);

        return iv;
    }
    vk::Sampler createTextureSampler(vk::Device device, vk::PhysicalDeviceProperties properties, uint32_t mip_levels)
    {
        vk::Sampler sampler;

        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.sType = vk::StructureType::eSamplerCreateInfo;
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
        samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = vk::CompareOp::eAlways;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.mipLodBias = 0.0f; //optional
        samplerInfo.minLod = 0.0f; //optional
        samplerInfo.maxLod = static_cast<float>(mip_levels);

        sampler = device.createSampler(samplerInfo);

        return sampler;
    }
	vk::Sampler createShadowSampler(vk::Device device, vk::PhysicalDeviceProperties properties)
	{
		vk::Sampler sampler;

		vk::SamplerCreateInfo samplerInfo{};
		samplerInfo.sType = vk::StructureType::eSamplerCreateInfo;
		samplerInfo.magFilter = vk::Filter::eLinear;
		samplerInfo.minFilter = vk::Filter::eLinear;
		samplerInfo.addressModeU = vk::SamplerAddressMode::eClampToEdge; // Change to ClampToEdge for shadow mapping
		samplerInfo.addressModeV = vk::SamplerAddressMode::eClampToEdge; // Change to ClampToEdge for shadow mapping
		samplerInfo.addressModeW = vk::SamplerAddressMode::eClampToEdge; // Change to ClampToEdge for shadow mapping
		samplerInfo.anisotropyEnable = VK_FALSE;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueBlack; // Change to FloatOpaqueBlack for better precision
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_TRUE; // Enable depth comparison
		samplerInfo.compareOp = vk::CompareOp::eLessOrEqual; // Compare operation for shadow mapping
		samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
		samplerInfo.mipLodBias = 0.0f; // Optional
		samplerInfo.minLod = 0.0f; // Optional
		samplerInfo.maxLod = 1.0f;

		sampler = device.createSampler(samplerInfo);


		return sampler;
	}
    void getTextureDimensions(const char* path, int& tex_width, int &tex_height)
    {
        int tex_channels;
        stbi_uc* pixels = stbi_load(path, &tex_width, &tex_height, &tex_channels, STBI_rgb_alpha);
        if (!pixels)
        {
            throw std::runtime_error("failed to load texture image!");
        }

        // You now have texWidth and texHeight, so you can free the pixels
        stbi_image_free(pixels);
    }
	stbi_uc* loadTexture(const char* path, int& width, int& height, int& channel, uint64_t& size)
	{
		stbi_uc* pixels = stbi_load(path, &width, &height, &channel, STBI_rgb_alpha);
		size = width * height * 4;
		if (!pixels)
			throw std::runtime_error("failed to load texture image!");

		return pixels;
	}





// DESCRIPTOR FUNCTIONS
	vk::DescriptorSetLayoutBinding createDescriptorSetLayoutBinding(uint8_t binding, vk::DescriptorType descriptor_type, vk::ShaderStageFlags shader_stage)
	{
		vk::DescriptorSetLayoutBinding layout_binding;
		layout_binding.binding = binding;
		layout_binding.descriptorCount = 1;
		layout_binding.descriptorType = descriptor_type;
		layout_binding.pImmutableSamplers = nullptr;
		layout_binding.stageFlags = shader_stage;

		return layout_binding;
	}

	vk::DescriptorSetLayout createDescriptorSetLayout(const vk::Device& device, std::vector<vk::DescriptorSetLayoutBinding> layout_bindings)
	{
		vk::DescriptorSetLayout desc_set_layout;

		vk::DescriptorSetLayoutCreateInfo layout_info{};
		layout_info.sType = vk::StructureType::eDescriptorSetLayoutCreateInfo;
		layout_info.bindingCount = layout_bindings.size();
		layout_info.pBindings = layout_bindings.data();

		try
		{
			desc_set_layout = device.createDescriptorSetLayout(layout_info);
		}
		catch (vk::SystemError err)
		{
			throw std::runtime_error("failed to create descriptor set layout!");
		}

		return desc_set_layout;
	}
	vk::DescriptorPoolSize createDescriptorPoolSize(vk::DescriptorType descriptor_type, uint32_t count)
	{
		vk::DescriptorPoolSize pool_size;
		pool_size.type = descriptor_type;
		pool_size.descriptorCount = count;
		
		return pool_size;
	}
	vk::DescriptorPool createDescriptorPool(vk::Device device, std::vector<vk::DescriptorPoolSize> pool_sizes, uint8_t max_sets)
	{
		vk::DescriptorPool pool;

		vk::DescriptorPoolCreateInfo pool_info = {};
		pool_info.sType = vk::StructureType::eDescriptorPoolCreateInfo;
		pool_info.poolSizeCount = pool_sizes.size();
		pool_info.pPoolSizes = pool_sizes.data();
		pool_info.maxSets = max_sets;

		pool = device.createDescriptorPool(pool_info);

		return pool;
	}
	vk::DescriptorSet allocateDescriptorSet(vk::Device device, vk::DescriptorSetLayout descriptor_set_layout, vk::DescriptorPool descriptor_pool)
	{
		vk::DescriptorSet descriptorSet;

		vk::DescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = vk::StructureType::eDescriptorSetAllocateInfo;
		allocInfo.descriptorPool = descriptor_pool;
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = &descriptor_set_layout;

		device.allocateDescriptorSets(&allocInfo, &descriptorSet);

		return descriptorSet;
	}


	vk::DescriptorImageInfo createDescriptorImageInfo(ImageDesc image_desc, vk::Sampler sampler, vk::ImageLayout layout)
	{
		vk::DescriptorImageInfo image_info{};
		image_info.imageLayout = layout;
		image_info.imageView = image_desc.imageview;
		image_info.sampler = sampler;

		return image_info;
	}
	vk::DescriptorBufferInfo createDescriptorBufferInfo(BufferDesc buffer_desc, uint64_t size)
	{
		vk::DescriptorBufferInfo buffer_info = {};
		buffer_info.buffer = buffer_desc.buffer;
		buffer_info.offset = 0;
		buffer_info.range = size;

		return buffer_info;
	}
	vk::WriteDescriptorSet createWriteDescriptorSet(vk::DescriptorSet descriptor_set, vk::DescriptorType type, vk::DescriptorBufferInfo buffer_info, uint32_t binding)
	{
		vk::WriteDescriptorSet write_descriptor_set;
		write_descriptor_set.sType = vk::StructureType::eWriteDescriptorSet;
		write_descriptor_set.dstSet = descriptor_set;
		write_descriptor_set.dstBinding = binding;
		write_descriptor_set.dstArrayElement = 0;
		write_descriptor_set.descriptorType = type;
		write_descriptor_set.descriptorCount = 1;
		write_descriptor_set.pBufferInfo = &buffer_info;

		return write_descriptor_set;
	}
	vk::WriteDescriptorSet createWriteDescriptorSet(vk::DescriptorSet descriptor_set, vk::DescriptorType type, vk::DescriptorImageInfo image_info, uint32_t binding)
	{
		vk::WriteDescriptorSet write_descriptor_set;
		write_descriptor_set.sType = vk::StructureType::eWriteDescriptorSet;
		write_descriptor_set.dstSet = descriptor_set;
		write_descriptor_set.dstBinding = binding;
		write_descriptor_set.dstArrayElement = 0;
		write_descriptor_set.descriptorType = type;
		write_descriptor_set.descriptorCount = 1;
		write_descriptor_set.pImageInfo = &image_info;

		return write_descriptor_set;
	}
	void writeToDescriptorSet(vk::Device device, std::vector<vk::WriteDescriptorSet> write_descriptor_sets)
	{
		device.updateDescriptorSets(2, write_descriptor_sets.data(), 0, nullptr);
	}





// DEVICE FUNCTIONS
	void findQueueFamilies(DeviceDesc& device_desc)
	{
		QueueFamilyIndices indices;

		uint32_t queue_family_count = 0;
		device_desc.physical_device.getQueueFamilyProperties(&queue_family_count, nullptr);

		std::vector<vk::QueueFamilyProperties> queue_families(queue_family_count);
		device_desc.physical_device.getQueueFamilyProperties(&queue_family_count, queue_families.data());
		std::cout << "\n\nYour device has " << queue_family_count << " Queue families" << "\n\n";
		device_desc.queue_families.resize(queue_family_count);
		device_desc.queue_families = queue_families;
	}
/*
		for (uint32_t i = 0; i < queue_family_count; i++)
		{
			std::cout << "\n" << static_cast<uint32_t>(queue_families[i].queueFlags) << "\n";
		}

		for (uint32_t i = 0; i < queue_family_count; i++)
		{
			if (queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics)
			{
				indices.graphics_family = i;
				std::cout << "\n Graphics -  " << i << "\n";
				break;
			}
		}

		vk::Bool32 present_support = false;
		for (uint32_t i = 0; i < queue_family_count; i++)
		{
			if (device_desc.device.getSurfaceSupportKHR(i, surface, &present_support) != vk::Result::eSuccess)
			{
				std::cout << "Failed to get Surface Support!\n";
			}

			if (present_support)
			{
				indices.present_family = i;
				std::cout << "\n Present -  " << i << "\n";
				break;
			}
		}

		for (uint32_t i = 0; i < queue_family_count; i++)
		{
			if (queue_families[i].queueFlags & vk::QueueFlagBits::eTransfer)
			{
				indices.transfer_family = i;
				std::cout << "\n Transfer -  " << i << "\n";
				break;
			}
		}

		for (uint32_t i = 0; i < queue_family_count; i++)
		{
			if (queue_families[i].queueFlags & vk::QueueFlagBits::eCompute)
			{
				indices.compute_family = i;
				std::cout << "\n Compute -  " << i << "\n";
				break;
			}
		}

        return indices;
    }
	*/
/*
	bool check_device_extension_support(vk::PhysicalDevice &physical_device)
	{
	    uint32_t extension_count;
	    if (physical_device.enumerateDeviceExtensionProperties(nullptr, &extension_count, nullptr) != vk::Result::eSuccess)
	    {
			std::cout << "Failed to get Device Extension Properties!\n";
	    }

	    std::vector<vk::ExtensionProperties> available_extensions(extension_count);
	    if (physical_device.enumerateDeviceExtensionProperties(nullptr, &extension_count, available_extensions.data()) != vk::Result::eSuccess)
	    {
			std::cout << "Failed to get Device Extension Properties!\n";
	    }

	    std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

	    for (const auto& extension : available_extensions)
	    {
	        required_extensions.erase(extension.extensionName);
	    }


	    return required_extensions.empty();
	}

/*
    bool is_device_suitable(vk::PhysicalDevice &physical_device)
    {
        QueueFamilyIndices indices = find_queue_families(physical_device);

	    bool extensions_supported = check_device_extension_support(physical_device);

		std::cout << "\n\n\n" << indices.graphics_family.has_value() << "  " << indices.present_family.has_value() << "\n\n";

        return indices.is_complete() && extensions_supported;
    }
*/





	void getPhysicalDevices(vk::Instance instance, std::vector<DeviceDesc> &device_descs)
	{
		std::vector<vk::PhysicalDevice> devices;
		uint32_t device_count = 0;
		if (instance.enumeratePhysicalDevices(&device_count, nullptr) != vk::Result::eSuccess)
		{
			std::cout << "Failed to enumerate Physical devices!\n";
		}

		if (device_count == 0)
		{
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		devices.resize(device_count);
		device_descs.resize(device_count);
		if (instance.enumeratePhysicalDevices(&device_count, devices.data())  != vk::Result::eSuccess)
		{
			std::cout << "Failed to enumerate Physical Devices!\n";
		}


		for (vk::PhysicalDevice device : devices)
		{
			vk::PhysicalDeviceProperties properties = device.getProperties();
			std::cout <<  "Device Name: " << properties.deviceName << "\n";
		}
		for (uint8_t i = 0; i < device_count; i++)
		{
			device_descs[i].physical_device = devices[i];
		}
	}


	vk::Device createLogicalDevice(DeviceDesc device_desc, const std::vector<const char*> device_extensions, const std::vector<const char*> validation_layers, bool enable_validation_layers)
	{
		vk::Device logical_device;

		std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;

		float queue_priority = 1.0f;
		for (uint8_t i = 0; i < device_desc.queue_families.size(); i++)
		{
			vk::DeviceQueueCreateInfo queue_create_info{};
			queue_create_info.sType = vk::StructureType::eDeviceQueueCreateInfo;
			queue_create_info.queueFamilyIndex = i;
			queue_create_info.queueCount = 1;
			queue_create_info.pQueuePriorities = &queue_priority;
			queue_create_infos.push_back(queue_create_info);
		}

		vk::PhysicalDeviceFeatures device_features{};

		vk::DeviceCreateInfo create_info{};
		create_info.sType = vk::StructureType::eDeviceCreateInfo;
		create_info.pQueueCreateInfos = queue_create_infos.data();
		create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
		create_info.pEnabledFeatures = &device_features;

		create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
		create_info.ppEnabledExtensionNames = device_extensions.data();

		if (enable_validation_layers)
		{
		    create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
		    create_info.ppEnabledLayerNames = validation_layers.data();
		}
		else
		{
		    create_info.enabledLayerCount = 0;
		}



		try
		{
			logical_device = device_desc.physical_device.createDevice(create_info);
		}
		catch (vk::SystemError err)
		{
			throw std::runtime_error("failed to create Logical Device");
		}

		return logical_device;
	}


	vk::SampleCountFlagBits getMaxUsableSampleCount(vk::PhysicalDevice physical_device)
	{
		vk::PhysicalDeviceProperties properties;
		physical_device.getProperties(&properties);

		vk::SampleCountFlags counts = properties.limits.framebufferColorSampleCounts & properties.limits.framebufferDepthSampleCounts;
		if (counts & vk::SampleCountFlagBits::e64) { return vk::SampleCountFlagBits::e64; }
		if (counts & vk::SampleCountFlagBits::e32) { return vk::SampleCountFlagBits::e32; }
		if (counts & vk::SampleCountFlagBits::e16) { return vk::SampleCountFlagBits::e16; }
		if (counts & vk::SampleCountFlagBits::e8)  { return vk::SampleCountFlagBits::e8;  }
		if (counts & vk::SampleCountFlagBits::e4)  { return vk::SampleCountFlagBits::e4;  }
		if (counts & vk::SampleCountFlagBits::e2)  { return vk::SampleCountFlagBits::e2;  }

		return vk::SampleCountFlagBits::e1;
	}




//	MATRIX CALCULATIONS
	void rotateModel(glm::mat4& modelMatrix, float angle, char axis)
	{
		glm::mat4 rotationMatrix = glm::mat4(1.0f); // Initialize with identity matrix

		// Convert angle from degrees to radians
		float rad = glm::radians(angle);

		// Apply rotation based on the axis
		switch (axis) {
		case 'x':
		case 'X':
			rotationMatrix = glm::rotate(rotationMatrix, rad, glm::vec3(1.0f, 0.0f, 0.0f));
			break;
		case 'y':
		case 'Y':
			rotationMatrix = glm::rotate(rotationMatrix, rad, glm::vec3(0.0f, 1.0f, 0.0f));
			break;
		case 'z':
		case 'Z':
			rotationMatrix = glm::rotate(rotationMatrix, rad, glm::vec3(0.0f, 0.0f, 1.0f));
			break;
		default:
			// Handle invalid axis
			std::cerr << "Invalid axis specified. Use 'x', 'y', or 'z'." << std::endl;
			return;
		}

		// Combine the new rotation with the existing model matrix
		modelMatrix = rotationMatrix * modelMatrix; // Pre-multiply to apply rotation in the correct order
	}


	void rotate(glm::mat4& model_matrix, float x, float y, float z)
	{
		glm::vec3 eular = glm::vec3(
			glm::radians(x),
			glm::radians(y),
			glm::radians(z)
		);
		glm::quat rotation = glm::quat(eular);

		glm::mat4 rotation_matrix = glm::toMat4(rotation);
		model_matrix = model_matrix * rotation_matrix;
	}

	void translate(glm::mat4& model, float x, float y, float z)
	{
		glm::mat4 identity = glm::mat4(1.0f);
		glm::mat4 translation_matrix = glm::translate(identity, glm::vec3(x, y, z));
		model = model * translation_matrix;
	}

	void scale(glm::mat4& model, float x, float y, float z)
	{
		glm::mat4 identity= glm::mat4(1.0f);
		glm::mat4 scale_matrix = glm::scale(identity, glm::vec3(x, y, z));
		model = model * scale_matrix;
	}



	void computePlaneTangents(
		std::vector<glm::vec3>& tangents,
		glm::vec3* positions,
		glm::vec2* texCoords,
		uint32_t quadVertexCount)
	{
		for (uint32_t i = 0; i < quadVertexCount; i += 3)
		{
			// Shortcuts for positions
			glm::vec3 pos0 = positions[i + 0];
			glm::vec3 pos1 = positions[i + 1];
			glm::vec3 pos2 = positions[i + 2];

			// Shortcuts for UVs
			glm::vec2 uv0 = texCoords[i + 0];
			glm::vec2 uv1 = texCoords[i + 1];
			glm::vec2 uv2 = texCoords[i + 2];

			// Edges of the triangle
			glm::vec3 edge1 = pos1 - pos0;
			glm::vec3 edge2 = pos2 - pos0;

			// UV delta
			glm::vec2 deltaUV1 = uv1 - uv0;
			glm::vec2 deltaUV2 = uv2 - uv0;

			// Compute the tangent
			float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

			glm::vec3 tangent;
			tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
			tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
			tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

			// Normalize the tangent vector
			tangent = glm::normalize(tangent);

			// Set the same tangent for all three vertices of the triangle
			tangents.push_back(tangent);
			tangents.push_back(tangent);
			tangents.push_back(tangent);
		}
	}

	void fillPlaneGrid(
		glm::vec3* positionPtr, glm::vec2* texCoordPtr, glm::vec3* normalPtr,
		uint32_t* indexPtr, glm::vec3 upVector, uint32_t n,
		std::vector<glm::vec3>& tangents, glm::vec3* positions, glm::vec2* texCoords, uint32_t quadVertexCount)
	{
		uint32_t vertexIndex = 0;

		for (unsigned int x = 0; x < n; x++)
		{
			for (unsigned int z = 0; z < n; z++)
			{
				positions[vertexIndex] = glm::vec3((z / (float)n) - 0.5f, 0, (x / (float)n) - 0.5f);
				texCoords[vertexIndex] = glm::vec2((z / (float)n), (x / (float)n));
				normalPtr[vertexIndex] = upVector;

				vertexIndex++;
			}
		}

		uint32_t indexCount = 0;

		for (unsigned int x = 0; x < n - 1; x++)
		{
			for (unsigned int z = 0; z < n - 1; z++)
			{
				indexPtr[indexCount++] = (x * n) + z;
				indexPtr[indexCount++] = ((x + 1) * n) + z;
				indexPtr[indexCount++] = ((x + 1) * n) + z + 1;

				indexPtr[indexCount++] = (x * n) + z;
				indexPtr[indexCount++] = ((x + 1) * n) + z + 1;
				indexPtr[indexCount++] = (x * n) + z + 1;
			}
		}

		// Compute tangents
		computePlaneTangents(tangents, positions, texCoords, quadVertexCount);
	}


	void computeTangents(const std::vector<glm::vec3>& positions, const std::vector<glm::vec2>& uv, const std::vector<uint32_t>& indices, std::vector<glm::vec3>& tangents)
	{
		tangents.resize(positions.size(), glm::vec3(0.0f));

		for (size_t i = 0; i < indices.size(); i += 3)
		{
			uint32_t i0 = indices[i + 0];
			uint32_t i1 = indices[i + 1];
			uint32_t i2 = indices[i + 2];

			glm::vec3 pos0 = positions[i0];
			glm::vec3 pos1 = positions[i1];
			glm::vec3 pos2 = positions[i2];

			glm::vec2 uv0 = uv[i0];
			glm::vec2 uv1 = uv[i1];
			glm::vec2 uv2 = uv[i2];

			glm::vec3 edge1 = pos1 - pos0;
			glm::vec3 edge2 = pos2 - pos0;

			glm::vec2 deltaUV1 = uv1 - uv0;
			glm::vec2 deltaUV2 = uv2 - uv0;

			float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

			glm::vec3 tangent;
			tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
			tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
			tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

			tangents[i0] += tangent;
			tangents[i1] += tangent;
			tangents[i2] += tangent;
		}

		// Normalize the tangents
		for (size_t i = 0; i < tangents.size(); ++i)
		{
			tangents[i] = glm::normalize(tangents[i]);
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

				indices.push_back(first);
				indices.push_back(second);
				indices.push_back(first + 1);

				indices.push_back(second);
				indices.push_back(second + 1);
				indices.push_back(first + 1);
			}
		}

		// Compute tangents
		computeTangents(positions, uv, indices, tangents);
	}


	// Function to convert equirectangular HDR to cube map
	void convertEquirectangularToCubemap(const float* hdrData, int width, int height, int faceSize, std::vector<float>& cubemapData)
	{
		cubemapData.resize(faceSize * faceSize * 6 * 4); // 6 faces, 4 channels (RGBA)

		const float PI = 3.14159265359f;
		const float TWO_PI = 2.0f * PI;

		for (int face = 0; face < 6; ++face) {
			for (int y = 0; y < faceSize; ++y) {
				for (int x = 0; x < faceSize; ++x) {
					float u = (x + 0.5f) / faceSize * 2.0f - 1.0f; // Convert to [-1, 1]
					float v = (y + 0.5f) / faceSize * 2.0f - 1.0f; // Convert to [-1, 1]

					// Compute direction vector
					glm::vec3 dir;
					if (face == 0) dir = glm::normalize(glm::vec3(1, v, -u));   // Positive X
					if (face == 1) dir = glm::normalize(glm::vec3(-1, v, u));   // Negative X
					if (face == 2) dir = glm::normalize(glm::vec3(u, -1, v));   // Positive Y (Top)
					if (face == 3) dir = glm::normalize(glm::vec3(u, 1, -v));   // Negative Y (Bottom)
					if (face == 4) dir = glm::normalize(glm::vec3(u, v, 1));    // Positive Z
					if (face == 5) dir = glm::normalize(glm::vec3(-u, v, -1));  // Negative Z

					// Convert direction vector to spherical coordinates
					float theta = atan2(dir.z, dir.x);
					float phi = acos(dir.y);
					float u_hdr = (theta + PI) / TWO_PI;
					float v_hdr = phi / PI;

					// Sample the HDR texture
					float sampleX = u_hdr * (width - 1);
					float sampleY = v_hdr * (height - 1);

					// Clamp coordinates (though not necessary since u_hdr and v_hdr are already in [0, 1])
					sampleX = fmaxf(0, fminf(sampleX, width - 1));
					sampleY = fmaxf(0, fminf(sampleY, height - 1));

					// Read pixel from HDR texture
					int pixelIndex = (static_cast<int>(sampleY) * width + static_cast<int>(sampleX)) * 3; // Assuming RGB format
					int cubemapIndex = (face * faceSize * faceSize + y * faceSize + x) * 4; // RGBA

					// Assign pixel values
					cubemapData[cubemapIndex] = hdrData[pixelIndex];       // R
					cubemapData[cubemapIndex + 1] = hdrData[pixelIndex + 1]; // G
					cubemapData[cubemapIndex + 2] = hdrData[pixelIndex + 2]; // B
					cubemapData[cubemapIndex + 3] = 1.0f; // A (constant alpha)
				}
			}
		}
	}

}