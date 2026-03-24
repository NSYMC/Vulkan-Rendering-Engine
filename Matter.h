#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include "Scene.h"

/**
 * Matter — Unity'deki Rigidbody benzeri fizik bileseni.
 * Bir SceneObject'e baglanir; kuvvet, agirlik, surtunme ile hareketi yonetir.
 * Her frame Step(deltaTime) cagrilarak pozisyon ve donus guncellenir.
 */
class Matter {
public:
	/** Hedef obje (nullptr ise Step etkisiz). */
	explicit Matter(SceneObject* target = nullptr);

	void SetTarget(SceneObject* target);
	SceneObject* GetTarget() const { return target_; }
	const SceneObject* GetTargetConst() const { return target_; }

	// ---- Kütle ----
	float GetMass() const { return mass_; }
	void SetMass(float mass);

	// ---- Hiz ve açısal hiz (m/s, rad/s) ----
	const glm::vec3& GetVelocity() const { return velocity_; }
	void SetVelocity(const glm::vec3& v);

	const glm::vec3& GetAngularVelocity() const { return angularVelocity_; }
	void SetAngularVelocity(const glm::vec3& omega);

	// ---- Kuvvet / tork (bir sonraki Step'te uygulanir; Step sonrasi sifirlanir) ----
	void AddForce(const glm::vec3& force);
	void AddForceAtPosition(const glm::vec3& force, const glm::vec3& worldPosition);
	void AddRelativeForce(const glm::vec3& force);
	void AddTorque(const glm::vec3& torque);

	// ---- Yerçekimi ----
	void SetUseGravity(bool use) { useGravity_ = use; }
	bool GetUseGravity() const { return useGravity_; }
	static void SetGravity(const glm::vec3& g) { GravityRef() = g; }
	static const glm::vec3& GetGravity() { return GravityRef(); }

	// ---- Surtunme (lineer / açısal) ----
	float GetDrag() const { return drag_; }
	void SetDrag(float d);
	float GetAngularDrag() const { return angularDrag_; }
	void SetAngularDrag(float d);

	// ---- Kinematik: true ise fizik entegrasyonu yapilmaz, sadece script ile hareket ----
	void SetKinematic(bool kinematic) { isKinematic_ = kinematic; }
	bool IsKinematic() const { return isKinematic_; }

	// ---- Entegrasyon: deltaTime (saniye) ile bir adim; hedef SceneObject guncellenir ----
	void Step(float deltaTime);

	// ---- Dogrudan hareket (kuvvet kullanmadan; kinematik veya script kontrolu) ----
	void MovePosition(const glm::vec3& worldPosition);
	void MoveRotation(const glm::vec3& eulerDegrees);
	void MovePositionDelta(const glm::vec3& delta);
	void MoveRotationDelta(const glm::vec3& eulerDegreesDelta);

	// ---- Pozisyon / donus kilidi (donus eksenleri derece cinsinden) ----
	void SetFreezePosition(bool x, bool y, bool z);
	void SetFreezeRotation(bool x, bool y, bool z);
	bool GetFreezePositionX() const { return freezePos_[0]; }
	bool GetFreezePositionY() const { return freezePos_[1]; }
	bool GetFreezePositionZ() const { return freezePos_[2]; }
	bool GetFreezeRotationX() const { return freezeRot_[0]; }
	bool GetFreezeRotationY() const { return freezeRot_[1]; }
	bool GetFreezeRotationZ() const { return freezeRot_[2]; }

	// ---- Atalet olcegi (tork -> acisal ivme: alpha = torque / (mass * inertiaScale)) ----
	float GetInertiaScale() const { return inertiaScale_; }
	void SetInertiaScale(float s) { inertiaScale_ = glm::max(0.001f, s); }

private:
	static glm::vec3& GravityRef() {
		static glm::vec3 g(0.0f, -9.81f, 0.0f);
		return g;
	}
	void SyncFromTarget();
	void ApplyVelocityConstraints();
	void ApplyAngularVelocityConstraints();

	SceneObject* target_ = nullptr;
	float mass_ = 1.0f;
	glm::vec3 velocity_ = glm::vec3(0.0f);
	glm::vec3 angularVelocity_ = glm::vec3(0.0f);  // rad/s (X,Y,Z eksenleri)
	glm::vec3 forceAccum_ = glm::vec3(0.0f);
	glm::vec3 torqueAccum_ = glm::vec3(0.0f);
	bool useGravity_ = true;
	float drag_ = 0.0f;
	float angularDrag_ = 0.05f;
	bool isKinematic_ = false;
	bool freezePos_[3] = { false, false, false };
	bool freezeRot_[3] = { false, false, false };
	float inertiaScale_ = 1.0f;
};

// ---------------------------------------------------------------------------
// Inline implementation
// ---------------------------------------------------------------------------

inline Matter::Matter(SceneObject* target) : target_(target) {}

inline void Matter::SetTarget(SceneObject* target) {
	target_ = target;
	if (target_) SyncFromTarget();
}

inline void Matter::SetMass(float mass) {
	mass_ = glm::max(0.001f, mass);
}

inline void Matter::SetVelocity(const glm::vec3& v) {
	velocity_ = v;
}

inline void Matter::SetAngularVelocity(const glm::vec3& omega) {
	angularVelocity_ = omega;
}

inline void Matter::SetDrag(float d) {
	drag_ = glm::max(0.0f, d);
}

inline void Matter::SetAngularDrag(float d) {
	angularDrag_ = glm::max(0.0f, d);
}

inline void Matter::AddForce(const glm::vec3& force) {
	forceAccum_ += force;
}

inline void Matter::AddTorque(const glm::vec3& torque) {
	torqueAccum_ += torque;
}

inline void Matter::AddForceAtPosition(const glm::vec3& force, const glm::vec3& worldPosition) {
	if (!target_) return;
	forceAccum_ += force;
	glm::vec3 center = target_->GetPosition();
	torqueAccum_ += glm::cross(worldPosition - center, force);
}

inline void Matter::AddRelativeForce(const glm::vec3& force) {
	if (!target_) return;
	glm::vec3 r = target_->GetRotation();
	float cy = std::cos(glm::radians(r.y)), sy = std::sin(glm::radians(r.y));
	float cx = std::cos(glm::radians(r.x)), sx = std::sin(glm::radians(r.x));
	float cz = std::cos(glm::radians(r.z)), sz = std::sin(glm::radians(r.z));
	glm::vec3 worldX(cy*cz - sy*sx*sz, cx*sz, -sy*cz - cy*sx*sz);
	glm::vec3 worldY(cy*sz + sy*sx*cz, cx*cz, -sy*sz + cy*sx*cz);
	glm::vec3 worldZ(sy*cx, -sx, cy*cx);
	glm::vec3 worldForce = worldX * force.x + worldY * force.y + worldZ * force.z;
	forceAccum_ += worldForce;
}

inline void Matter::SetFreezePosition(bool x, bool y, bool z) {
	freezePos_[0] = x; freezePos_[1] = y; freezePos_[2] = z;
}

inline void Matter::SetFreezeRotation(bool x, bool y, bool z) {
	freezeRot_[0] = x; freezeRot_[1] = y; freezeRot_[2] = z;
}

inline void Matter::SyncFromTarget() {
	if (!target_ || target_->IsDestroyed()) return;
	velocity_ = glm::vec3(0.0f);
	angularVelocity_ = glm::vec3(0.0f);
	forceAccum_ = glm::vec3(0.0f);
	torqueAccum_ = glm::vec3(0.0f);
}

inline void Matter::ApplyVelocityConstraints() {
	if (freezePos_[0]) velocity_.x = 0.0f;
	if (freezePos_[1]) velocity_.y = 0.0f;
	if (freezePos_[2]) velocity_.z = 0.0f;
}

inline void Matter::ApplyAngularVelocityConstraints() {
	if (freezeRot_[0]) angularVelocity_.x = 0.0f;
	if (freezeRot_[1]) angularVelocity_.y = 0.0f;
	if (freezeRot_[2]) angularVelocity_.z = 0.0f;
}

inline void Matter::Step(float deltaTime) {
	if (!target_ || target_->IsDestroyed() || deltaTime <= 0.0f) return;

	if (isKinematic_) {
		forceAccum_ = glm::vec3(0.0f);
		torqueAccum_ = glm::vec3(0.0f);
		return;
	}

	float m = mass_;
	if (m < 1e-6f) m = 1e-6f;

	// Yerçekimi
	if (useGravity_)
		forceAccum_ += GravityRef() * m;

	// Lineer surtunme (F = -drag * v)
	forceAccum_ -= drag_ * velocity_;
	glm::vec3 acceleration = forceAccum_ / m;
	velocity_ += acceleration * deltaTime;
	ApplyVelocityConstraints();

	// Açısal surtunme ve tork entegrasyonu
	torqueAccum_ -= angularDrag_ * angularVelocity_;
	float invAngularMass = 1.0f / (m * inertiaScale_);
	angularVelocity_ += torqueAccum_ * invAngularMass * deltaTime;
	ApplyAngularVelocityConstraints();

	// Pozisyon guncelle (derece/s yerine rad/s kullanildigi icin donusum)
	glm::vec3 pos = target_->GetPosition();
	pos += velocity_ * deltaTime;
	target_->SetPosition(pos);

	glm::vec3 rot = target_->GetRotation();
	rot += glm::degrees(angularVelocity_) * deltaTime;
	target_->SetRotation(rot);

	forceAccum_ = glm::vec3(0.0f);
	torqueAccum_ = glm::vec3(0.0f);
}

inline void Matter::MovePosition(const glm::vec3& worldPosition) {
	if (target_ && !target_->IsDestroyed())
		target_->SetPosition(worldPosition);
}

inline void Matter::MoveRotation(const glm::vec3& eulerDegrees) {
	if (target_ && !target_->IsDestroyed())
		target_->SetRotation(eulerDegrees);
}

inline void Matter::MovePositionDelta(const glm::vec3& delta) {
	if (target_ && !target_->IsDestroyed())
		target_->SetPosition(target_->GetPosition() + delta);
}

inline void Matter::MoveRotationDelta(const glm::vec3& eulerDegreesDelta) {
	if (target_ && !target_->IsDestroyed())
		target_->SetRotation(target_->GetRotation() + eulerDegreesDelta);
}
