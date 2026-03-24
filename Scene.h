#pragma once

#include <vector>
#include <list>
#include <memory>
#include <unordered_map>
#include <functional>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Utilities.h"

class Scene;

/**
 * Sahnedeki tek bir obje (Unity'deki GameObject benzeri).
 * Spawn edilen her mesh objesi bu sinifin bir ornegidir.
 * Pozisyon, rotasyon (X,Y,Z derece), scale manupile edilebilir; Destroy() ile kaldirilir.
 */
class SceneObject {
public:
	// ---- Transform ----
	const glm::vec3& GetPosition() const { return position_; }
	void SetPosition(const glm::vec3& pos) { position_ = pos; }

	/** Euler acilari (derece): X = pitch, Y = yaw, Z = roll. Her eksende donus. */
	const glm::vec3& GetRotation() const { return rotation_; }
	void SetRotation(const glm::vec3& eulerDegrees) { rotation_ = eulerDegrees; }

	const glm::vec3& GetScale() const { return scale_; }
	void SetScale(const glm::vec3& s) { scale_ = s; }

	int GetModelId() const { return modelId_; }

	/** Dunya model matrisi: Scale * Rotation(YXZ) * Translation */
	glm::mat4 GetModelMatrix() const;

	/** Objeyi sahneden kaldirir ve cizimden cikarir. Cagrildiktan sonra bu pointer kullanilmayacak. */
	void Destroy();

	bool IsDestroyed() const { return destroyed_; }

private:
	SceneObject(Scene* scene, size_t id, int modelId, const glm::vec3& position, const glm::vec3& rotationDegrees, const glm::vec3& scale);
	Scene* scene_ = nullptr;
	size_t myId_ = 0;
	int modelId_ = 0;
	glm::vec3 position_;
	glm::vec3 rotation_;  // Euler derece (X, Y, Z)
	glm::vec3 scale_;
	bool destroyed_ = false;

	friend class Scene;
};

// ---------------------------------------------------------------------------
// Scene: SceneObject'lari yonetir, isiklar ayri tutulur.
// ---------------------------------------------------------------------------
class Scene {
public:
	using OnObjectDestroyed = std::function<void(int modelId)>;

	Scene() = default;
	~Scene() = default;

	/** Obje yok edildiginde cagrilir (renderer gizleme, panel refresh). */
	void SetOnObjectDestroyed(OnObjectDestroyed cb) { onObjectDestroyed_ = std::move(cb); }

	// ---- SceneObject'lar (spawn / erisim / yok etme) ----
	/** Yeni obje olusturur; spawn edilen her obje bu class'a sahiptir. Donen pointer Destroy() cagrilana kadar gecerli. */
	SceneObject* CreateObject(int modelId, const glm::vec3& position = glm::vec3(0.0f),
		const glm::vec3& rotationDegrees = glm::vec3(0.0f), const glm::vec3& scale = glm::vec3(1.0f));

	size_t GetObjectCount() const { return objects_.size(); }

	/** i. objeye pointer (0 <= i < GetObjectCount()). Destroy sonrasi indeksler kayar, pointer saklamak tehlikeli. */
	SceneObject* GetObjectAt(size_t i);

	const SceneObject* GetObjectAt(size_t i) const;

	/** i. objenin model matrisi (GetObjectAt(i)->GetModelMatrix() ile ayni). */
	glm::mat4 GetModelMatrixAt(size_t i) const;

	/** Indeks ile kaldirma (panel "Kaldir" icin). Iceride GetObjectAt(i)->Destroy() cagirir. */
	void RemoveObjectAt(size_t i);

	// ---- Eski API uyumluluk (index ile; iceride GetObjectAt kullanir) ----
	void setObjectPosition(size_t index, const glm::vec3& position) {
		SceneObject* o = GetObjectAt(index);
		if (o) o->SetPosition(position);
	}

	// ---- Point lights ----
	size_t getLightCount() const { return lights.size(); }
	const UserPointLight& getLight(size_t index) const { return lights[index]; }
	UserPointLight& getLightMutable(size_t index) { return lights[index]; }
	void setLight(size_t index, const UserPointLight& L) {
		if (index < lights.size()) lights[index] = L;
	}
	bool addLight(const UserPointLight& L) {
		if (lights.size() >= (size_t)MAX_POINT_LIGHTS) return false;
		lights.push_back(L);
		return true;
	}
	void removeLight(size_t index) {
		if (index < lights.size()) lights.erase(lights.begin() + (ptrdiff_t)index);
	}

	// ---- Spot lights ----
	size_t getSpotLightCount() const { return spotLights.size(); }
	const UserSpotLight& getSpotLight(size_t index) const { return spotLights[index]; }
	UserSpotLight& getSpotLightMutable(size_t index) { return spotLights[index]; }
	void setSpotLight(size_t index, const UserSpotLight& L) {
		if (index < spotLights.size()) spotLights[index] = L;
	}
	bool addSpotLight(const UserSpotLight& L) {
		if (spotLights.size() >= (size_t)MAX_SPOT_LIGHTS) return false;
		spotLights.push_back(L);
		return true;
	}
	void removeSpotLight(size_t index) {
		if (index < spotLights.size()) spotLights.erase(spotLights.begin() + (ptrdiff_t)index);
	}

	void clear() {
		objects_.clear();
		idToIterator_.clear();
		nextId_ = 0;
		lights.clear();
		spotLights.clear();
	}

	/** Id ile objeyi bulup sahneden kaldirir (SceneObject::Destroy icinden cagrilir). */
	void DestroyObjectById(size_t id);

private:
	using ObjectList = std::list<std::unique_ptr<SceneObject>>;
	ObjectList objects_;
	std::unordered_map<size_t, ObjectList::iterator> idToIterator_;
	size_t nextId_ = 0;
	OnObjectDestroyed onObjectDestroyed_;

	std::vector<UserPointLight> lights;
	std::vector<UserSpotLight> spotLights;
};

// ---------------------------------------------------------------------------
// SceneObject inline impl
// ---------------------------------------------------------------------------
inline glm::mat4 SceneObject::GetModelMatrix() const {
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, position_);
	// YXZ: once Y (yaw), sonra X (pitch), sonra Z (roll) - glm::rotate ile deneysel extension gerektirmez
	model = glm::rotate(model, glm::radians(rotation_.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotation_.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(rotation_.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::scale(model, scale_);
	return model;
}

inline void SceneObject::Destroy() {
	if (destroyed_ || !scene_) return;
	destroyed_ = true;
	scene_->DestroyObjectById(myId_);
}

inline SceneObject::SceneObject(Scene* scene, size_t id, int modelId, const glm::vec3& position, const glm::vec3& rotationDegrees, const glm::vec3& scale)
	: scene_(scene), myId_(id), modelId_(modelId), position_(position), rotation_(rotationDegrees), scale_(scale) {}

// ---------------------------------------------------------------------------
// Scene inline impl
// ---------------------------------------------------------------------------
inline SceneObject* Scene::CreateObject(int modelId, const glm::vec3& position, const glm::vec3& rotationDegrees, const glm::vec3& scale) {
	size_t id = nextId_++;
	SceneObject* raw = new SceneObject(this, id, modelId, position, rotationDegrees, scale);
	objects_.push_back(std::unique_ptr<SceneObject>(raw));
	idToIterator_[id] = std::prev(objects_.end());
	return raw;
}

inline SceneObject* Scene::GetObjectAt(size_t i) {
	if (i >= objects_.size()) return nullptr;
	auto it = objects_.begin();
	std::advance(it, (ptrdiff_t)i);
	return it->get();
}

inline const SceneObject* Scene::GetObjectAt(size_t i) const {
	if (i >= objects_.size()) return nullptr;
	auto it = objects_.begin();
	std::advance(it, (ptrdiff_t)i);
	return it->get();
}

inline glm::mat4 Scene::GetModelMatrixAt(size_t i) const {
	const SceneObject* o = GetObjectAt(i);
	return o ? o->GetModelMatrix() : glm::mat4(1.0f);
}

inline void Scene::RemoveObjectAt(size_t i) {
	SceneObject* o = GetObjectAt(i);
	if (o) o->Destroy();
}

inline void Scene::DestroyObjectById(size_t id) {
	auto fit = idToIterator_.find(id);
	if (fit == idToIterator_.end()) return;
	ObjectList::iterator lit = fit->second;
	int modelId = (*lit)->GetModelId();
	objects_.erase(lit);
	idToIterator_.erase(fit);
	if (onObjectDestroyed_) onObjectDestroyed_(modelId);
}
