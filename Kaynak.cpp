#define STB_IMAGE_IMPLEMENTATION

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <stdexcept>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <windows.h>
#include <glm/glm.hpp>
#include "VulkanRender.h"
#include "Input.h"
#include "Camera.h"
#include "Scene.h"
#include "Matter.h"
#include "ModelPanel.h"
#include "LightPanel.h"
#include "ScenePanel.h"
#include "Utilities.h"

VulkanRender vulkanRenderer;
Input input;
Camera camera;
GLFWwindow* window;
float deltaTime;
float angle;





void setWindow(std::string wName = "Test window", const int width = 1920, const int height = 1080) {
	glfwInit();

	// says glfw to not work with opengl
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(width, height, wName.c_str(),nullptr,nullptr);
}

int main() {


	if (LoadLibraryA("C:\\Program Files\\RenderDoc\\renderdoc.dll")) {
		std::cout << "RenderDoc basariyla kancalandi!" << std::endl;
	}
	else {
		std::cout << "RenderDoc DLL bulunamadi. Yol dogru mu kontrol et." << std::endl;
	}


	


	setWindow();
	camera.input = input;


	
	if (vulkanRenderer.init(window) == EXIT_FAILURE) {
		return EXIT_FAILURE;
	}

	angle = 0.0f;
	deltaTime = 0.0f;
	float lastTime = 0.0f;

	/* true = sunum: F1 + beyaz zemin + golge | false = Dunya + Ay + F1 yorungesi */
	const bool sunumGolgeDemo = true;
	const glm::vec3 sceneCenter(0.0f, 0.0f, 0.0f);
	glm::vec3 starPosition(25.0f, 0.0f, 0.0f);
	const float starGravityStrength = 25.0f;
	const float minDistToStar = 5.0f;
	const float earthOrbitRadius = 25.0f;
	const float earthMass = 1.0f;
	const float moonOrbitRadius = 5.0f;
	const float moonOrbitSpeedRadPerSec = 0.4f;
	const float f1OrbitRadius = 10.0f;
	const float f1OrbitSpeedRadPerSec = 0.8f;

	Scene scene;
	int floorModelId = -1;
	SceneObject* earthObj = nullptr;
	Matter* pEarthMatter = nullptr;
	SceneObject* moonObj = nullptr;
	float moonOrbitAngle = 0.0f;
	SceneObject* f1Obj = nullptr;
	float f1OrbitAngle = 0.0f;

	if (sunumGolgeDemo) {
		int whiteTex = vulkanRenderer.createSolidColorTexture(255, 255, 255, 255);
		floorModelId = vulkanRenderer.createFloorMeshModel(55.0f, whiteTex);
		vulkanRenderer.setFloorModelId(floorModelId);
		starPosition = glm::vec3(14.0f, 22.0f, 11.0f);
		try {
			int f1ModelId = vulkanRenderer.createMeshModel("Models/mcl35m_2.obj");
			f1Obj = scene.CreateObject(f1ModelId, glm::vec3(0.0f, 0.06f, 0.0f), glm::vec3(0.0f, 40.0f, 0.0f), glm::vec3(0.28f));
		} catch (const std::exception& e) {
			std::cerr << "F1 yuklenemedi: " << e.what() << std::endl;
		}
		std::cout << "[SUNUM] F5=normal | F6=UV (golge tex adresi) | F7=sert golge | F8=derinlik kontrast\n";
	} else {
		vulkanRenderer.setFloorModelId(-1);
		int earthModelId = vulkanRenderer.createMeshModel("EarthObject/scene.gltf");
		earthObj = scene.CreateObject(earthModelId, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f), glm::vec3(1.0f));
		pEarthMatter = new Matter(earthObj);
		pEarthMatter->SetMass(earthMass);
		pEarthMatter->SetUseGravity(false);
		pEarthMatter->SetDrag(0.0f);
		float earthOrbitSpeed = std::sqrt(starGravityStrength / (earthMass * earthOrbitRadius));
		pEarthMatter->SetVelocity(glm::vec3(0.0f, 0.0f, earthOrbitSpeed));
		try {
			int moonModelId = vulkanRenderer.createMeshModel("MoonObject/scene.gltf");
			moonObj = scene.CreateObject(moonModelId, glm::vec3(moonOrbitRadius, 0.0f, 0.0f), glm::vec3(0.0f), glm::vec3(0.5f));
		} catch (const std::exception& e) {
			std::cerr << "Ay yuklenemedi: " << e.what() << std::endl;
		}
		try {
			int f1ModelId = vulkanRenderer.createMeshModel("Models/mcl35m_2.obj");
			f1Obj = scene.CreateObject(f1ModelId, starPosition + glm::vec3(f1OrbitRadius, 0.0f, 0.0f), glm::vec3(0.0f), glm::vec3(0.4f));
		} catch (const std::exception& e) {
			std::cerr << "F1 yuklenemedi: " << e.what() << std::endl;
		}
	}

	// Sahne isiksiz (point/spot eklenmez); directional gunes yine var (gormek icin)

	glfwSetWindowTitle(window, sunumGolgeDemo ? "VulkanApp | SUNUM F1+golge (F5-F7)" : "VulkanApp");
	int wndW = 0, wndH = 0;
	glfwGetWindowSize(window, &wndW, &wndH);
	int modelAddCount = 0;
	ModelPanel modelPanel;
	ScenePanel scenePanel;
	modelPanel.create(glfwGetWin32Window(window), wndW, wndH,
		[&](const std::string& path) {
			try {
				int id = vulkanRenderer.createMeshModel(path);
				float offset = (float)modelAddCount * 4.0f;
				scene.CreateObject(id, glm::vec3(offset, 10.0f, 5.0f), glm::vec3(0.0f), glm::vec3(1.0f));
				modelAddCount++;
				scenePanel.refreshList();
			} catch (const std::exception&) {}
		});

	
	glfwSetKeyCallback(window, Input::KeyCallback);
	glfwSetCursorPosCallback(window, Input::cursor_positin_callback);
	glfwSetMouseButtonCallback(window, Input::mouse_button_callback);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	camera.SetWindow(window);

	// ---- Directional light = yildiz (golge de yildizdan; sunlight gecici kapali) ----
	LightUbo lightData = {};
	lightData.sunParams = glm::vec4(1.2f, 0.35f, 1.0f, 0.15f);   // intensity, ambientStrength, shadowBiasScale, shadowDarken
	lightData.shadowParams = glm::vec4(2.0f, 0.0f, 0.0f, 0.0f);  // x=softness, y=debug (0/1/2)

	// Blackbody: sicaklik (K) -> RGB (yildiz frag shader ile ayni formül)
	auto blackbody = [](float T) -> glm::vec3 {
		T = std::max(1000.f, std::min(40000.f, T));
		float t = T / 100.f;
		float r, g, b;
		if (t <= 66.f) {
			r = 1.f;
			g = std::max(0.f, std::min(1.f, 0.390081578f * std::log(std::max(t, 0.01f)) - 0.631841443f));
			b = t <= 19.f ? 0.f : std::max(0.f, std::min(1.f, 0.543206789f * std::log(std::max(t - 10.f, 0.01f)) - 1.196254089f));
		} else {
			r = std::max(0.f, std::min(1.f, 1.292936186f * std::pow(t - 60.f, -0.1332047592f)));
			g = std::max(0.f, std::min(1.f, 1.129890861f * std::pow(t - 60.f, -0.0755148492f)));
			b = 1.f;
		}
		return glm::vec3(r, g, b);
	};

	// Isiklar sahnede tutulur; L ile isik panelinden eklenir, O ile sol listeden kaldirilabilir

	LightPanel lightPanel;
	lightPanel.create(glfwGetWin32Window(window), wndW, wndH, &scene, [&]() { scenePanel.refreshList(); });

	scene.SetOnObjectDestroyed([&](int modelId) {
		vulkanRenderer.setModelVisible(modelId, false);
		lightPanel.refreshList();
		scenePanel.refreshList();
	});

	scenePanel.create(glfwGetWin32Window(window), wndW, wndH, &scene, [&](int) {
		lightPanel.refreshList();
	});

	// Obje secimi ve surukleme (gizmo + pick)
	int selectedObjectIndex = -1;
	scenePanel.setSelectionAfterRemove(&selectedObjectIndex);
	int selectedGizmoAxis = 0;  // 0=serbest, 1=X, 2=Y, 3=Z (sadece o eksende tasi)
	const float pickRadius = 3.0f;  // bounding sphere yaricapi
	float lastMouseX = 0.0f, lastMouseY = 0.0f;
	bool wasLeftDown = false;
	glm::vec3 lastWorldPosOnPlane(0.0f);
	glm::vec3 dragPlaneOrigin(0.0f), dragPlaneNormal(0.0f, 0.0f, -1.0f);  // surukleme basinda sabit duzlem (obje derinligi, 1:1 hiz)

	auto raySphereHit = [](const glm::vec3& rayOrig, const glm::vec3& rayDir, const glm::vec3& center, float radius) -> float {
		glm::vec3 A = rayOrig - center;
		float a = glm::dot(rayDir, rayDir);
		float b = 2.0f * glm::dot(A, rayDir);
		float c = glm::dot(A, A) - radius * radius;
		float disc = b * b - 4.0f * a * c;
		if (disc < 0.0f) return -1.0f;
		float sqrtDisc = std::sqrt(disc);
		float t0 = (-b - sqrtDisc) / (2.0f * a);
		float t1 = (-b + sqrtDisc) / (2.0f * a);
		if (t0 > 0.0f) return t0;
		if (t1 > 0.0f) return t1;
		return -1.0f;
	};
	// Earth kendi etrafinda donus hizi (derece/saniye). Bu degiskeni degistirerek hizi ayarlayabilirsin.
	float earthRotationSpeedDegPerSec = 20.0f;

	// Yildiz konumu: ustte glm::vec3 starPosition; sunum modunda (14,22,11), tam sahnede (25,0,0)
	float starTemperatureKelvin = 5800.0f;  // 3000=Kirmizi cuce, 5800=Gunes, 15000+=Mavi dev

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		float now = glfwGetTime();
		deltaTime = now - lastTime;
		lastTime = now;

		if (input.GetKey(GLFW_KEY_E)) angle += 10.f * deltaTime;
		if (input.GetKey(GLFW_KEY_Q)) angle -= 10.f * deltaTime;

		if (input.GetMouseButton(GLFW_MOUSE_BUTTON_RIGHT))
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		else
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

		// Directional isik = yildiz (fragment'ta L = normalize(starPosition - worldPos) icin starPosition gonder)
		glm::vec3 toStar = starPosition - sceneCenter;
		if (glm::length(toStar) > 1e-5f)
			lightData.direction = glm::vec4(glm::normalize(toStar), 0.0f);
		lightData.color = glm::vec4(blackbody(starTemperatureKelvin), 1.0f);
		lightData.starPosition = glm::vec4(starPosition, 1.0f);  // w=1 => shader L from worldPos to star

		// Point lights: sahnedeki isiklar -> UBO (L panelinden eklenir)
		size_t nLights = std::min(scene.getLightCount(), (size_t)MAX_POINT_LIGHTS);
		lightData.numPointLights = (int)nLights;
		for (size_t i = 0; i < nLights; i++) {
			const UserPointLight& L = scene.getLight(i);
			lightData.pointLights[i].position = glm::vec4(L.position, L.enabled ? 1.0f : 0.0f);
			lightData.pointLights[i].color    = glm::vec4(glm::max(L.color, glm::vec3(0.0f)), L.intensity);
			lightData.pointLights[i].params   = glm::vec4(std::max(L.range, 0.001f), 0.0f, 0.0f, 0.0f);
		}
		for (size_t i = nLights; i < (size_t)MAX_POINT_LIGHTS; i++) {
			lightData.pointLights[i].position.w = 0.0f;
		}

		// Spot lights: sahnedeki spot isiklar -> UBO
		size_t nSpot = std::min(scene.getSpotLightCount(), (size_t)MAX_SPOT_LIGHTS);
		lightData.numSpotLights = (int)nSpot;
		for (size_t i = 0; i < nSpot; i++) {
			const UserSpotLight& S = scene.getSpotLight(i);
			glm::vec3 dir = glm::length(S.direction) > 1e-6f ? glm::normalize(S.direction) : glm::vec3(0.f, -1.f, 0.f);
			float innerRad = glm::radians(std::max(0.1f, S.innerAngleDeg));
			float outerRad = glm::radians(std::max(S.innerAngleDeg, S.outerAngleDeg));
			lightData.spotLights[i].position  = glm::vec4(S.position, S.enabled ? 1.0f : 0.0f);
			lightData.spotLights[i].direction = glm::vec4(dir, 0.0f);
			lightData.spotLights[i].color     = glm::vec4(glm::max(S.color, glm::vec3(0.0f)), S.intensity);
			lightData.spotLights[i].params     = glm::vec4(std::max(S.range, 0.001f), std::cos(innerRad), std::cos(outerRad), 0.0f);
		}
		for (size_t i = nSpot; i < (size_t)MAX_SPOT_LIGHTS; i++) {
			lightData.spotLights[i].position.w = 0.0f;
		}

		camera.Update(deltaTime);
		vulkanRenderer.cam = camera.GetViewMatrix();
		// Golge kamera sahne merkezi = Dunya (yildizdan Dunya'ya bakis; Ay'in Dunya uzerine golgesi mantikli duser)
		vulkanRenderer.shadowSceneCenter = sunumGolgeDemo ? glm::vec3(0.0f)
			: ((earthObj && !earthObj->IsDestroyed()) ? earthObj->GetPosition() : sceneCenter);

		if (input.GetKeyDown(GLFW_KEY_F5))
			lightData.shadowParams.y = 0.0f;
		if (input.GetKeyDown(GLFW_KEY_F6))
			lightData.shadowParams.y = 1.0f;  // UV haritasi
		if (input.GetKeyDown(GLFW_KEY_F7))
			lightData.shadowParams.y = 2.0f;  // sert golge
		if (input.GetKeyDown(GLFW_KEY_F8))
			lightData.shadowParams.y = 3.0f;  // derinlik kontrast

		vulkanRenderer.updateLight(lightData);

		// Obje secimi: sol tikla pick, surukle tasi (gizmo + konum guncellemesi bu blokta)
		int scW = 0, scH = 0;
		vulkanRenderer.getExtent(scW, scH);
		int wndW = 0, wndH = 0;
		glfwGetWindowSize(window, &wndW, &wndH);
		// Mouse pencere koordinatinda; NDC icin viewport (swapchain) uzayina cevir (DPI uyumu, yesil ok Y)
		float mx = wndW > 0 ? (float)Input::mouseX * ((float)scW / (float)wndW) : (float)Input::mouseX;
		float my = wndH > 0 ? (float)Input::mouseY * ((float)scH / (float)wndH) : (float)Input::mouseY;
		bool leftDown = input.GetMouseButton(GLFW_MOUSE_BUTTON_LEFT);

		if (leftDown && !wasLeftDown) {
			// Yeni tik: once gizmo ekseni (secili obje varsa), yoksa obje pick
			glm::vec3 rayOrig, rayDir;
			vulkanRenderer.getPickRay(mx, my, (float)scW, (float)scH, rayOrig, rayDir);
			int axisHit = 0;
			if (selectedObjectIndex >= 0) {
				SceneObject* sel = scene.GetObjectAt(selectedObjectIndex);
				if (sel)
					axisHit = vulkanRenderer.pickGizmoAxis(mx, my, (float)scW, (float)scH, sel->GetPosition(), camera.GetViewMatrix(), vulkanRenderer.getProjectionMatrix());
			}
			if (axisHit >= 1 && selectedObjectIndex >= 0) {
				SceneObject* sel = scene.GetObjectAt(selectedObjectIndex);
				if (sel) {
					selectedGizmoAxis = axisHit;
					dragPlaneOrigin = sel->GetPosition();
					dragPlaneNormal = vulkanRenderer.getCameraForward();
					lastWorldPosOnPlane = vulkanRenderer.getWorldPosOnPlane(mx, my, (float)scW, (float)scH, dragPlaneOrigin, dragPlaneNormal);
				}
			} else {
				selectedGizmoAxis = 0;
				float bestT = 1e30f;
				int hitIndex = -1;
				for (size_t i = 0; i < scene.GetObjectCount(); i++) {
					SceneObject* o = scene.GetObjectAt(i);
					if (!o) continue;
					float t = raySphereHit(rayOrig, rayDir, o->GetPosition(), pickRadius);
					if (t > 0.0f && t < bestT) { bestT = t; hitIndex = (int)i; }
				}
				selectedObjectIndex = hitIndex;
				if (selectedObjectIndex >= 0) {
					vulkanRenderer.setLightGizmoSelected(true);
					lastWorldPosOnPlane = vulkanRenderer.getWorldPosOnNearPlane(mx, my, (float)scW, (float)scH);
				} else
					vulkanRenderer.setLightGizmoSelected(false);
			}
		} else if (!leftDown) {
			selectedGizmoAxis = 0;
			if (selectedObjectIndex < 0)
				vulkanRenderer.setLightGizmoSelected(false);
		}

		// Tutulan ekseni goster (parlak ok). Hareket SADECE ok seciliyken (tek yol)
		vulkanRenderer.setGizmoHighlightAxis(selectedGizmoAxis);
		if (leftDown && selectedObjectIndex >= 0 && selectedGizmoAxis >= 1) {
			SceneObject* sel = scene.GetObjectAt(selectedObjectIndex);
			if (sel) {
				glm::vec3 curWorld = vulkanRenderer.getWorldPosOnPlane(mx, my, (float)scW, (float)scH, dragPlaneOrigin, dragPlaneNormal);
				glm::vec3 delta = curWorld - lastWorldPosOnPlane;
				glm::vec3 newPos;
				if (selectedGizmoAxis == 1)
					newPos = sel->GetPosition() + glm::vec3(glm::dot(delta, glm::vec3(1,0,0)), 0.0f, 0.0f);
				else if (selectedGizmoAxis == 2)
					newPos = sel->GetPosition() + glm::vec3(0.0f, glm::dot(delta, glm::vec3(0,1,0)), 0.0f);
				else
					newPos = sel->GetPosition() + glm::vec3(0.0f, 0.0f, glm::dot(delta, glm::vec3(0,0,1)));
				sel->SetPosition(newPos);
				glm::vec3 sc = sel->GetScale();
				float objRad = std::max({ sc.x, sc.y, sc.z });
				vulkanRenderer.updateGizmoBuffer(newPos, objRad);
				lastWorldPosOnPlane = curWorld;
			}
		} else if (selectedObjectIndex >= 0) {
			SceneObject* sel = scene.GetObjectAt(selectedObjectIndex);
			if (sel) {
				glm::vec3 sc = sel->GetScale();
				float objRad = std::max({ sc.x, sc.y, sc.z });
				vulkanRenderer.updateGizmoBuffer(sel->GetPosition(), objRad);
			}
		}

		wasLeftDown = leftDown;
		lastMouseX = mx;
		lastMouseY = my;

		// Fizik: sadece Dunya (Gunes cekimi); kucuk adimlarla
		const float physicsDtMax = 1.0f / 120.0f;
		float remaining = std::min(deltaTime, 0.25f);
		if (!sunumGolgeDemo) {
			while (remaining > 1e-6f) {
				float dt = std::min(remaining, physicsDtMax);
				remaining -= dt;
				if (pEarthMatter && earthObj && !earthObj->IsDestroyed()) {
					glm::vec3 earthPos = earthObj->GetPosition();
					glm::vec3 toStar = starPosition - earthPos;
					float r = std::max(glm::length(toStar), minDistToStar);
					pEarthMatter->AddForce((toStar / r) * (starGravityStrength / (r * r)));
					pEarthMatter->Step(dt);
				}
			}
			if (moonObj && !moonObj->IsDestroyed() && earthObj && !earthObj->IsDestroyed()) {
				moonOrbitAngle += moonOrbitSpeedRadPerSec * deltaTime;
				glm::vec3 earthPos = earthObj->GetPosition();
				glm::vec3 offset(moonOrbitRadius * std::cos(moonOrbitAngle), 0.0f, moonOrbitRadius * std::sin(moonOrbitAngle));
				moonObj->SetPosition(earthPos + offset);
			}
			if (f1Obj && !f1Obj->IsDestroyed()) {
				f1OrbitAngle += f1OrbitSpeedRadPerSec * deltaTime;
				glm::vec3 offset(f1OrbitRadius * std::cos(f1OrbitAngle), 0.0f, f1OrbitRadius * std::sin(f1OrbitAngle));
				f1Obj->SetPosition(starPosition + offset);
			}
		}

		if (sunumGolgeDemo && floorModelId >= 0)
			vulkanRenderer.updateModel(floorModelId, glm::mat4(1.0f));

		for (size_t i = 0; i < scene.GetObjectCount(); i++) {
			SceneObject* o = scene.GetObjectAt(i);
			if (!o) continue;
			glm::mat4 modelMat = o->GetModelMatrix();
			vulkanRenderer.updateModel(o->GetModelId(), modelMat);
		}

		if (input.GetKeyDown(GLFW_KEY_M))
			modelPanel.toggleVisibility();
		if (input.GetKeyDown(GLFW_KEY_L))
			lightPanel.toggleVisibility();
		if (input.GetKeyDown(GLFW_KEY_O))
			scenePanel.toggleVisibility();

		ModelPanel::pollMessages();
		ScenePanel::pollMessages();

		// Yildiz: sicakliga gore yaricap (3000K kucuk, 5800K=1x Gunes, 20000K buyuk)
		float T = std::max(3000.f, std::min(30000.f, starTemperatureKelvin));
		float starRadius;
		if (T <= 5800.f)
			starRadius = 0.35f + (T - 3000.f) * (1.0f - 0.35f) / (5800.f - 3000.f);
		else if (T <= 20000.f)
			starRadius = 1.0f + (T - 5800.f) * (2.0f - 1.0f) / (20000.f - 5800.f);
		else
			starRadius = 2.0f + (T - 20000.f) * 0.5f / 10000.f;
		glm::mat4 starModel = glm::translate(glm::mat4(1.f), starPosition) * glm::scale(glm::mat4(1.f), glm::vec3(starRadius));
		vulkanRenderer.setStarParams(starModel, static_cast<float>(now), starTemperatureKelvin);

		vulkanRenderer.draw();
		Input::Update();
	}

	if (pEarthMatter)
		delete pEarthMatter;

	scenePanel.destroy();
	lightPanel.destroy();
	modelPanel.destroy();
	vulkanRenderer.cleanup();

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}




