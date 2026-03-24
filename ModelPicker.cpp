#include "ModelPicker.h"
#include "VulkanRender.h"
#include "Scene.h"
#include "Input.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <windows.h>
#include <algorithm>

std::vector<std::string> getObjPathsInFolder(const std::string& folderPath) {
	std::vector<std::string> paths;
	std::string searchPath = folderPath;
	if (searchPath.back() != '/' && searchPath.back() != '\\')
		searchPath += "\\";
	searchPath += "*.obj";

	WIN32_FIND_DATAA findData;
	HANDLE hFind = FindFirstFileA(searchPath.c_str(), &findData);
	if (hFind == INVALID_HANDLE_VALUE)
		return paths;

	do {
		if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
			std::string name(findData.cFileName);
			paths.push_back(folderPath + "/" + name);
		}
	} while (FindNextFileA(hFind, &findData));
	FindClose(hFind);

	std::sort(paths.begin(), paths.end());
	return paths;
}

ModelPickerUI::ModelPickerUI(VulkanRender* renderer, Scene* scene, GLFWwindow* window)
	: renderer_(renderer), scene_(scene), window_(window) {
	const char* title = glfwGetWindowTitle(window);
	defaultTitle_ = title ? title : "VulkanApp";
	refreshModelList();
}

void ModelPickerUI::refreshModelList() {
	modelPaths_ = getObjPathsInFolder("Models");
}

void ModelPickerUI::updateWindowTitle() {
	if (!window_) return;
	if (!menuVisible_) {
		glfwSetWindowTitle(window_, defaultTitle_.c_str());
		return;
	}
	std::string title = "[M] Kapat | Sahneye ekle: ";
	for (size_t i = 0; i < modelPaths_.size() && i < 9; i++) {
		size_t slash = modelPaths_[i].find_last_of("/\\");
		std::string name = slash != std::string::npos ? modelPaths_[i].substr(slash + 1) : modelPaths_[i];
		title += std::to_string(i + 1) + "=" + name;
		if (i + 1 < modelPaths_.size() && i + 1 < 9) title += " ";
	}
	glfwSetWindowTitle(window_, title.c_str());
}

void ModelPickerUI::update(Input& input) {
	if (input.GetKeyDown(GLFW_KEY_M)) {
		menuVisible_ = !menuVisible_;
		updateWindowTitle();
	}

	if (!menuVisible_ || modelPaths_.empty()) return;

	for (int i = 0; i < 9 && i < (int)modelPaths_.size(); i++) {
		int key = GLFW_KEY_1 + i;
		if (input.GetKeyDown(key)) {
			try {
				int modelId = renderer_->createMeshModel(modelPaths_[i]);
				float offset = (float)addedCount_ * 4.0f;
				glm::vec3 pos(offset, 0.0f, 0.0f);
				scene_->addObject(modelId, pos, 0.0f, glm::vec3(1.0f));
				addedCount_++;
			}
			catch (const std::exception& e) {
				// Model load failed; could log to console
			}
			break;
		}
	}
}
