


namespace gltfModel
{
	void ExtractMeshData(const tinygltf::Model& model, std::vector<float>& vertices, std::vector<uint32_t>& indices)
	{
	    for (const auto& mesh : model.meshes)
	    {
	        for (const auto& primitive : mesh.primitives)
	        {
	            const auto& posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
	            const auto& posBufferView = model.bufferViews[posAccessor.bufferView];
	            const auto& posBuffer = model.buffers[posBufferView.buffer];

	            const float* positions = reinterpret_cast<const float*>(
	                posBuffer.data.data() + posBufferView.byteOffset + posAccessor.byteOffset
	            );

	            vertices.insert(vertices.end(), positions, positions + posAccessor.count * 3);

	            // Extract indices
	            if (primitive.indices >= 0)
	            {
	                const auto& indexAccessor = model.accessors[primitive.indices];
	                const auto& indexBufferView = model.bufferViews[indexAccessor.bufferView];
	                const auto& indexBuffer = model.buffers[indexBufferView.buffer];

	                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
	                {
	                    const uint16_t* buffer = reinterpret_cast<const uint16_t*>(
	                        indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset
	                    );
	                    for (size_t i = 0; i < indexAccessor.count; i++)
	                    {
	                        indices.push_back(buffer[i]);
	                    }
	                }
	                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
	                {
	                    const uint32_t* buffer = reinterpret_cast<const uint32_t*>(
	                        indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset
	                    );
	                    indices.insert(indices.end(), buffer, buffer + indexAccessor.count);
	                }
	            }
	        }
	    }
	}

}