#include "LightPanel.h"
#include "Utilities.h"
#include "Scene.h"
#include <glm/glm.hpp>
#include <windows.h>
#include <commctrl.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

#pragma comment(lib, "comctl32.lib")

namespace {
	void setEditFloat(HWND h, float value) {
		char buf[32];
		snprintf(buf, sizeof(buf), "%.3f", value);
		SetWindowTextA(h, buf);
	}
	float getEditFloat(HWND h, float defaultVal) {
		char buf[32];
		if (GetWindowTextA(h, buf, (int)sizeof(buf)) <= 0) return defaultVal;
		float v = defaultVal;
#ifdef _MSC_VER
		sscanf_s(buf, "%f", &v);
#else
		sscanf(buf, "%f", &v);
#endif
		return v;
	}
}

LightPanel* g_lightPanelInstance = nullptr;

LRESULT CALLBACK LightPanel::PanelWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
	LightPanel* self = g_lightPanelInstance;
	if (!self || self->hwnd_ != hwnd)
		return DefWindowProcA(hwnd, msg, wParam, lParam);

	switch (msg) {
	case WM_COMMAND: {
		WORD id = LOWORD(wParam);
		WORD notify = HIWORD(wParam);
		if (id == IDC_ADD_POINT && self->scene_) {
			UserPointLight def = { glm::vec3(2.f, 2.f, 2.f), glm::vec3(1.f, 1.f, 1.f), 1.f, 10.f, true };
			if (self->scene_->addLight(def)) {
				self->refreshComboFromScene();
				self->selectedIndex_ = (int)self->scene_->getLightCount() - 1;
				SendMessageA(self->combo_, CB_SETCURSEL, self->selectedIndex_, 0);
				self->refreshEditControls();
				if (self->onLightListChanged_) self->onLightListChanged_();
			}
		}
		else if (notify == CBN_SELCHANGE && id == IDC_COMBO) {
			self->applyFromEditControls();
			self->selectedIndex_ = (int)SendMessageA(self->combo_, CB_GETCURSEL, 0, 0);
			if (self->selectedIndex_ < 0) self->selectedIndex_ = 0;
			self->refreshEditControls();
		}
		else if (id == IDC_ENABLED && self->scene_ && self->selectedIndex_ >= 0 && self->selectedIndex_ < (int)self->scene_->getLightCount())
			self->scene_->getLightMutable((size_t)self->selectedIndex_).enabled = (SendMessageA(self->chkEnabled_, BM_GETCHECK, 0, 0) == BST_CHECKED);
		else if (notify == EN_CHANGE && self->scene_ && self->selectedIndex_ >= 0 && self->selectedIndex_ < (int)self->scene_->getLightCount()) {
			UserPointLight& L = self->scene_->getLightMutable((size_t)self->selectedIndex_);
			L.position.x = getEditFloat(self->editPX_, L.position.x);
			L.position.y = getEditFloat(self->editPY_, L.position.y);
			L.position.z = getEditFloat(self->editPZ_, L.position.z);
			L.color.r = getEditFloat(self->editCR_, L.color.r);
			L.color.g = getEditFloat(self->editCG_, L.color.g);
			L.color.b = getEditFloat(self->editCB_, L.color.b);
			L.intensity = getEditFloat(self->editInt_, L.intensity);
			L.range = getEditFloat(self->editRange_, L.range);
		}
		if (id == IDC_ADD_SPOT && self->scene_) {
			UserSpotLight def = { glm::vec3(0.f, 3.f, 0.f), glm::vec3(0.f, -1.f, 0.f), glm::vec3(1.f, 1.f, 1.f), 1.2f, 15.f, 25.f, 45.f, true };
			if (self->scene_->addSpotLight(def)) {
				self->refreshSpotComboFromScene();
				self->selectedSpotIndex_ = (int)self->scene_->getSpotLightCount() - 1;
				SendMessageA(self->comboSpot_, CB_SETCURSEL, self->selectedSpotIndex_, 0);
				self->refreshSpotEditControls();
				if (self->onLightListChanged_) self->onLightListChanged_();
			}
		}
		else if (notify == CBN_SELCHANGE && id == IDC_COMBO_SPOT) {
			self->applyFromSpotEditControls();
			self->selectedSpotIndex_ = (int)SendMessageA(self->comboSpot_, CB_GETCURSEL, 0, 0);
			if (self->selectedSpotIndex_ < 0) self->selectedSpotIndex_ = 0;
			self->refreshSpotEditControls();
		}
		else if (id == IDC_ENABLED_SPOT && self->scene_ && self->selectedSpotIndex_ >= 0 && self->selectedSpotIndex_ < (int)self->scene_->getSpotLightCount())
			self->scene_->getSpotLightMutable((size_t)self->selectedSpotIndex_).enabled = (SendMessageA(self->chkEnabledSpot_, BM_GETCHECK, 0, 0) == BST_CHECKED);
		else if (notify == EN_CHANGE && self->scene_ && self->selectedSpotIndex_ >= 0 && self->selectedSpotIndex_ < (int)self->scene_->getSpotLightCount()) {
			UserSpotLight& S = self->scene_->getSpotLightMutable((size_t)self->selectedSpotIndex_);
			S.position.x = getEditFloat(self->editSpotPX_, S.position.x);
			S.position.y = getEditFloat(self->editSpotPY_, S.position.y);
			S.position.z = getEditFloat(self->editSpotPZ_, S.position.z);
			S.direction.x = getEditFloat(self->editSpotDX_, S.direction.x);
			S.direction.y = getEditFloat(self->editSpotDY_, S.direction.y);
			S.direction.z = getEditFloat(self->editSpotDZ_, S.direction.z);
			S.color.r = getEditFloat(self->editSpotCR_, S.color.r);
			S.color.g = getEditFloat(self->editSpotCG_, S.color.g);
			S.color.b = getEditFloat(self->editSpotCB_, S.color.b);
			S.intensity = getEditFloat(self->editSpotInt_, S.intensity);
			S.range = getEditFloat(self->editSpotRange_, S.range);
			S.innerAngleDeg = getEditFloat(self->editSpotInner_, S.innerAngleDeg);
			S.outerAngleDeg = getEditFloat(self->editSpotOuter_, S.outerAngleDeg);
		}
		break;
	}
	case WM_DESTROY:
		self->hwnd_ = nullptr;
		self->combo_ = nullptr;
		self->btnAddPoint_ = nullptr;
		self->btnAddSpot_ = nullptr;
		self->chkEnabled_ = nullptr;
		self->editPX_ = self->editPY_ = self->editPZ_ = nullptr;
		self->editCR_ = self->editCG_ = self->editCB_ = nullptr;
		self->editInt_ = self->editRange_ = nullptr;
		self->comboSpot_ = nullptr;
		self->chkEnabledSpot_ = nullptr;
		self->editSpotPX_ = self->editSpotPY_ = self->editSpotPZ_ = nullptr;
		self->editSpotDX_ = self->editSpotDY_ = self->editSpotDZ_ = nullptr;
		self->editSpotCR_ = self->editSpotCG_ = self->editSpotCB_ = nullptr;
		self->editSpotInt_ = self->editSpotRange_ = self->editSpotInner_ = self->editSpotOuter_ = nullptr;
		self->scene_ = nullptr;
		self->onLightListChanged_ = nullptr;
		break;
	default:
		return DefWindowProcA(hwnd, msg, wParam, lParam);
	}
	return 0;
}

LightPanel::LightPanel() {
	g_lightPanelInstance = this;
}

LightPanel::~LightPanel() {
	destroy();
	g_lightPanelInstance = nullptr;
}

void LightPanel::create(void* mainWindowHWND, int mainWidth, int mainHeight, Scene* scene, OnLightListChanged onLightListChanged) {
	if (hwnd_) return;
	mainWindowHWND_ = (HWND)mainWindowHWND;
	scene_ = scene;
	onLightListChanged_ = std::move(onLightListChanged);
	selectedIndex_ = 0;

	WNDCLASSEXA wc = {};
	wc.cbSize = sizeof(wc);
	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.lpfnWndProc = PanelWndProc;
	wc.hInstance = GetModuleHandle(nullptr);
	wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
	wc.lpszClassName = "LightPanelClass";
	if (!RegisterClassExA(&wc))
		throw std::runtime_error("LightPanel: RegisterClassEx failed");

	HMONITOR hMon = MonitorFromWindow(mainWindowHWND_, MONITOR_DEFAULTTONEAREST);
	MONITORINFO mi = {};
	mi.cbSize = sizeof(mi);
	GetMonitorInfoA(hMon, &mi);
	workArea_ = mi.rcWork;
	int workW = workArea_.right - workArea_.left;
	int workH = workArea_.bottom - workArea_.top;
	int panelH = panelHeight_;
	int contentH = workH - panelH;
	if (contentH < 300) contentH = workH / 2;

	int x = workArea_.left;
	int y = workArea_.top + contentH;
	int w = workW;
	int h = panelH;

	hwnd_ = CreateWindowExA(0, "LightPanelClass", "Isiklar (L) - Ekle / duzenle",
		WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
		x, y, w, h, nullptr, nullptr, wc.hInstance, nullptr);
	if (!hwnd_)
		throw std::runtime_error("LightPanel: CreateWindow failed");
	visible_ = false;
	ShowWindow(hwnd_, SW_HIDE);

	int left = 10, row = 8, editW = 52, editH = 18, gap = 4, labelW = 28;
	auto mkEdit = [&](int id, int xpos, int ypos) -> HWND {
		return CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
			WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
			xpos, ypos, editW, editH, hwnd_, (HMENU)(INT_PTR)id, wc.hInstance, nullptr);
	};
	auto mkLabel = [&](const char* text, int xpos, int ypos) {
		CreateWindowExA(0, "STATIC", text, WS_CHILD | WS_VISIBLE, xpos, ypos, labelW, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	};

	btnAddPoint_ = CreateWindowExA(0, "BUTTON", "Point Isik Ekle", WS_CHILD | WS_VISIBLE,
		left, row - 2, 120, editH + 4, hwnd_, (HMENU)IDC_ADD_POINT, wc.hInstance, nullptr);
	left += 128;

	CreateWindowExA(0, "STATIC", "Point:", WS_CHILD | WS_VISIBLE, left, row, 32, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	combo_ = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
		left + 36, row - 2, 100, 120, hwnd_, (HMENU)IDC_COMBO, wc.hInstance, nullptr);
	left += 150;

	chkEnabled_ = CreateWindowExA(0, "BUTTON", "Acik", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, left, row - 2, 50, editH + 2, hwnd_, (HMENU)IDC_ENABLED, wc.hInstance, nullptr);
	left += 60;

	refreshComboFromScene();

	row += editH + 8;
	left = 10;
	mkLabel("Pos X", left, row); editPX_ = mkEdit(IDC_PX, left + labelW, row); left += labelW + editW + gap;
	mkLabel("Y", left, row);     editPY_ = mkEdit(IDC_PY, left + 18, row);   left += 18 + editW + gap;
	mkLabel("Z", left, row);     editPZ_ = mkEdit(IDC_PZ, left + 18, row);   left += 18 + editW + gap;
	row += editH + 6;
	left = 10;
	mkLabel("R", left, row);     editCR_ = mkEdit(IDC_CR, left + 18, row);   left += 18 + editW + gap;
	mkLabel("G", left, row);     editCG_ = mkEdit(IDC_CG, left + 18, row);   left += 18 + editW + gap;
	mkLabel("B", left, row);     editCB_ = mkEdit(IDC_CB, left + 18, row);   left += 18 + editW + gap;
	row += editH + 6;
	left = 10;
	CreateWindowExA(0, "STATIC", "Intensity", WS_CHILD | WS_VISIBLE, left, row, 50, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	editInt_ = mkEdit(IDC_INT, left + 52, row); left += 52 + editW + 20;
	CreateWindowExA(0, "STATIC", "Range", WS_CHILD | WS_VISIBLE, left, row, 36, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	editRange_ = mkEdit(IDC_RANGE, left + 40, row);

	row += editH + 14;
	left = 10;
	btnAddSpot_ = CreateWindowExA(0, "BUTTON", "Spot Isik Ekle", WS_CHILD | WS_VISIBLE,
		left, row - 2, 120, editH + 4, hwnd_, (HMENU)IDC_ADD_SPOT, wc.hInstance, nullptr);
	left += 128;
	CreateWindowExA(0, "STATIC", "Spot:", WS_CHILD | WS_VISIBLE, left, row, 32, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	comboSpot_ = CreateWindowExA(0, "COMBOBOX", "", WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
		left + 36, row - 2, 100, 120, hwnd_, (HMENU)IDC_COMBO_SPOT, wc.hInstance, nullptr);
	left += 150;
	chkEnabledSpot_ = CreateWindowExA(0, "BUTTON", "Acik", WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, left, row - 2, 50, editH + 2, hwnd_, (HMENU)IDC_ENABLED_SPOT, wc.hInstance, nullptr);
	refreshSpotComboFromScene();

	row += editH + 8;
	left = 10;
	mkLabel("Pos", left, row); editSpotPX_ = mkEdit(IDC_SPX, left + 28, row); left += 28 + editW + gap;
	mkLabel("Y", left, row); editSpotPY_ = mkEdit(IDC_SPY, left + 18, row);   left += 18 + editW + gap;
	mkLabel("Z", left, row); editSpotPZ_ = mkEdit(IDC_SPZ, left + 18, row);   left += 18 + editW + gap;
	mkLabel("Dir", left + 4, row); editSpotDX_ = mkEdit(IDC_SDX, left + 28, row); left += 28 + editW + gap;
	editSpotDY_ = mkEdit(IDC_SDY, left, row); left += editW + gap;
	editSpotDZ_ = mkEdit(IDC_SDZ, left, row);
	row += editH + 6;
	left = 10;
	mkLabel("R", left, row); editSpotCR_ = mkEdit(IDC_SCR, left + 18, row); left += 18 + editW + gap;
	mkLabel("G", left, row); editSpotCG_ = mkEdit(IDC_SCG, left + 18, row); left += 18 + editW + gap;
	mkLabel("B", left, row); editSpotCB_ = mkEdit(IDC_SCB, left + 18, row); left += 18 + editW + gap;
	CreateWindowExA(0, "STATIC", "Int", WS_CHILD | WS_VISIBLE, left + 4, row, 18, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	editSpotInt_ = mkEdit(IDC_SINT, left + 26, row); left += 26 + editW + 4;
	CreateWindowExA(0, "STATIC", "Range", WS_CHILD | WS_VISIBLE, left, row, 28, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	editSpotRange_ = mkEdit(IDC_SRANGE, left + 32, row); left += 32 + editW + 4;
	CreateWindowExA(0, "STATIC", "In", WS_CHILD | WS_VISIBLE, left, row, 14, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	editSpotInner_ = mkEdit(IDC_SINNER, left + 18, row); left += 18 + editW + 2;
	CreateWindowExA(0, "STATIC", "Out", WS_CHILD | WS_VISIBLE, left, row, 18, editH, hwnd_, nullptr, wc.hInstance, nullptr);
	editSpotOuter_ = mkEdit(IDC_SOUTER, left + 22, row);

	refreshEditControls();
	refreshSpotEditControls();
}

void LightPanel::destroy() {
	if (hwnd_) {
		DestroyWindow(hwnd_);
		hwnd_ = nullptr;
		combo_ = nullptr;
		btnAddPoint_ = btnAddSpot_ = nullptr;
		chkEnabled_ = chkEnabledSpot_ = nullptr;
		editPX_ = editPY_ = editPZ_ = nullptr;
		editCR_ = editCG_ = editCB_ = nullptr;
		editInt_ = editRange_ = nullptr;
		comboSpot_ = nullptr;
		editSpotPX_ = editSpotPY_ = editSpotPZ_ = nullptr;
		editSpotDX_ = editSpotDY_ = editSpotDZ_ = nullptr;
		editSpotCR_ = editSpotCG_ = editSpotCB_ = nullptr;
		editSpotInt_ = editSpotRange_ = editSpotInner_ = editSpotOuter_ = nullptr;
		scene_ = nullptr;
		onLightListChanged_ = nullptr;
		mainWindowHWND_ = nullptr;
	}
	UnregisterClassA("LightPanelClass", GetModuleHandle(nullptr));
}

void LightPanel::refreshComboFromScene() {
	if (!combo_ || !scene_) return;
	SendMessageA(combo_, CB_RESETCONTENT, 0, 0);
	size_t n = scene_->getLightCount();
	for (size_t i = 0; i < n; i++) {
		char buf[32];
		snprintf(buf, sizeof(buf), "Point Isik %zu", i);
		SendMessageA(combo_, CB_ADDSTRING, 0, (LPARAM)buf);
	}
	if (n > 0) {
		if (selectedIndex_ < 0 || selectedIndex_ >= (int)n) selectedIndex_ = 0;
		SendMessageA(combo_, CB_SETCURSEL, selectedIndex_, 0);
	}
}

void LightPanel::refreshSpotComboFromScene() {
	if (!comboSpot_ || !scene_) return;
	SendMessageA(comboSpot_, CB_RESETCONTENT, 0, 0);
	size_t n = scene_->getSpotLightCount();
	for (size_t i = 0; i < n; i++) {
		char buf[32];
		snprintf(buf, sizeof(buf), "Spot Isik %zu", i);
		SendMessageA(comboSpot_, CB_ADDSTRING, 0, (LPARAM)buf);
	}
	if (n > 0) {
		if (selectedSpotIndex_ < 0 || selectedSpotIndex_ >= (int)n) selectedSpotIndex_ = 0;
		SendMessageA(comboSpot_, CB_SETCURSEL, selectedSpotIndex_, 0);
	}
}

void LightPanel::refreshList() {
	refreshComboFromScene();
	refreshSpotComboFromScene();
	refreshEditControls();
	refreshSpotEditControls();
}

void LightPanel::refreshEditControls() {
	if (!scene_ || selectedIndex_ < 0 || (size_t)selectedIndex_ >= scene_->getLightCount()) return;
	const UserPointLight& L = scene_->getLight((size_t)selectedIndex_);
	SendMessageA(chkEnabled_, BM_SETCHECK, L.enabled ? BST_CHECKED : BST_UNCHECKED, 0);
	setEditFloat(editPX_, L.position.x);
	setEditFloat(editPY_, L.position.y);
	setEditFloat(editPZ_, L.position.z);
	setEditFloat(editCR_, L.color.r);
	setEditFloat(editCG_, L.color.g);
	setEditFloat(editCB_, L.color.b);
	setEditFloat(editInt_, L.intensity);
	setEditFloat(editRange_, L.range);
}

void LightPanel::applyFromEditControls() {
	if (!scene_ || selectedIndex_ < 0 || (size_t)selectedIndex_ >= scene_->getLightCount()) return;
	UserPointLight& L = scene_->getLightMutable((size_t)selectedIndex_);
	L.enabled = (SendMessageA(chkEnabled_, BM_GETCHECK, 0, 0) == BST_CHECKED);
	L.position.x = getEditFloat(editPX_, L.position.x);
	L.position.y = getEditFloat(editPY_, L.position.y);
	L.position.z = getEditFloat(editPZ_, L.position.z);
	L.color.r = getEditFloat(editCR_, L.color.r);
	L.color.g = getEditFloat(editCG_, L.color.g);
	L.color.b = getEditFloat(editCB_, L.color.b);
	L.intensity = getEditFloat(editInt_, L.intensity);
	L.range = getEditFloat(editRange_, L.range);
}

void LightPanel::refreshSpotEditControls() {
	if (!scene_ || selectedSpotIndex_ < 0 || (size_t)selectedSpotIndex_ >= scene_->getSpotLightCount()) return;
	const UserSpotLight& S = scene_->getSpotLight((size_t)selectedSpotIndex_);
	SendMessageA(chkEnabledSpot_, BM_SETCHECK, S.enabled ? BST_CHECKED : BST_UNCHECKED, 0);
	setEditFloat(editSpotPX_, S.position.x);
	setEditFloat(editSpotPY_, S.position.y);
	setEditFloat(editSpotPZ_, S.position.z);
	setEditFloat(editSpotDX_, S.direction.x);
	setEditFloat(editSpotDY_, S.direction.y);
	setEditFloat(editSpotDZ_, S.direction.z);
	setEditFloat(editSpotCR_, S.color.r);
	setEditFloat(editSpotCG_, S.color.g);
	setEditFloat(editSpotCB_, S.color.b);
	setEditFloat(editSpotInt_, S.intensity);
	setEditFloat(editSpotRange_, S.range);
	setEditFloat(editSpotInner_, S.innerAngleDeg);
	setEditFloat(editSpotOuter_, S.outerAngleDeg);
}

void LightPanel::applyFromSpotEditControls() {
	if (!scene_ || selectedSpotIndex_ < 0 || (size_t)selectedSpotIndex_ >= scene_->getSpotLightCount()) return;
	UserSpotLight& S = scene_->getSpotLightMutable((size_t)selectedSpotIndex_);
	S.enabled = (SendMessageA(chkEnabledSpot_, BM_GETCHECK, 0, 0) == BST_CHECKED);
	S.position.x = getEditFloat(editSpotPX_, S.position.x);
	S.position.y = getEditFloat(editSpotPY_, S.position.y);
	S.position.z = getEditFloat(editSpotPZ_, S.position.z);
	S.direction.x = getEditFloat(editSpotDX_, S.direction.x);
	S.direction.y = getEditFloat(editSpotDY_, S.direction.y);
	S.direction.z = getEditFloat(editSpotDZ_, S.direction.z);
	S.color.r = getEditFloat(editSpotCR_, S.color.r);
	S.color.g = getEditFloat(editSpotCG_, S.color.g);
	S.color.b = getEditFloat(editSpotCB_, S.color.b);
	S.intensity = getEditFloat(editSpotInt_, S.intensity);
	S.range = getEditFloat(editSpotRange_, S.range);
	S.innerAngleDeg = getEditFloat(editSpotInner_, S.innerAngleDeg);
	S.outerAngleDeg = getEditFloat(editSpotOuter_, S.outerAngleDeg);
}

void LightPanel::setPositionBelow(int mainX, int mainY, int mainWidth, int mainHeight) {
	if (!hwnd_) return;
	SetWindowPos(hwnd_, nullptr, mainX, mainY + mainHeight, mainWidth, panelHeight_, SWP_NOZORDER | SWP_NOACTIVATE);
}

void LightPanel::updateLayout(bool panelVisible) {
	if (!hwnd_ || !mainWindowHWND_) return;
	int workW = workArea_.right - workArea_.left;
	int workH = workArea_.bottom - workArea_.top;
	int panelH = panelHeight_;
	if (panelVisible) {
		SetWindowPos(hwnd_, nullptr, workArea_.left, workArea_.top + workH - panelH, workW, panelH, SWP_NOZORDER);
		ShowWindow(hwnd_, SW_SHOW);
		SetWindowPos(hwnd_, HWND_TOP, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
	}
	else
		ShowWindow(hwnd_, SW_HIDE);
}

void LightPanel::toggleVisibility() {
	if (!hwnd_) return;
	visible_ = !visible_;
	updateLayout(visible_);
}

void LightPanel::pollMessages() {
	MSG msg;
	while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
		TranslateMessage(&msg);
		DispatchMessageA(&msg);
	}
}
