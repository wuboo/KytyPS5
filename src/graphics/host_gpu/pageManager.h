#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_

#include "common/common.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <memory>

namespace Libs::Graphics {

enum class PageFaultAccess { Read, Write, Execute, Unknown };

class PageManager final {
public:
	PageManager();
	// The owner must stop all PageManager callers before destruction.
	~PageManager();

	KYTY_CLASS_NO_COPY(PageManager);

	[[nodiscard]] uint64_t GetPageSize() const;

	template <bool track>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size);
	template <bool track, bool is_read = false>
	void UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask);

	// End of the run of consecutive watched (protected) pages that starts at vaddr's page,
	// scanning at most max_bytes. Returns the page start of vaddr when that page is unwatched.
	[[nodiscard]] uint64_t WatchedRunEnd(uint64_t vaddr, uint64_t max_bytes) const noexcept;

	// A read fault the caches could not satisfy (no GPU data to download). When the tracker
	// state of vaddr's page allows reads but the host page is not readable, set the protection
	// the tracker expects (Read or ReadWrite) and return true; *previous_host_protection gets
	// the host VM_PROT bits seen before (-1 when the platform cannot tell). Returns false when
	// nothing was changed (page unknown, tracker wants no access, or already readable).
	bool RestoreReadAccess(uint64_t vaddr, int* previous_host_protection = nullptr);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PAGEMANAGER_H_
