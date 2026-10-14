// The dialog shell the planner-style modals share (task plan, add reference, the
// ControlNet guide) and the input-file <select> they fill.

let plannerModalSerial = 0;

export function createPlannerModal(title, extraClass = "") {
    document.querySelectorAll(".mmrp-overlay.mmrp-planner-overlay").forEach((el) => {
        el.querySelector(".mmrp-modal")?.dispatchEvent(new Event("mmrp-modal-close"));
        el.remove();
    });
    const previouslyFocused = document.activeElement;
    const overlay = document.createElement("div");
    overlay.className = "mmrp-overlay mmrp-planner-overlay";
    const modal = document.createElement("div");
    modal.className = `mmrp-modal mmrp-planner-modal ${extraClass}`.trim();
    const header = document.createElement("div");
    header.className = "mmrp-modal-header mmrp-planner-header";
    const heading = document.createElement("span");
    heading.id = `mmrp-planner-title-${++plannerModalSerial}`;
    heading.textContent = title;
    const closeButton = document.createElement("button");
    closeButton.className = "mmrp-btn mmrp-planner-close";
    closeButton.textContent = "×";
    closeButton.setAttribute("aria-label", "Close");
    header.appendChild(heading);
    header.appendChild(closeButton);
    modal.appendChild(header);
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.setAttribute("aria-labelledby", heading.id);
    overlay.appendChild(modal);

    const onKey = (event) => {
        if (event.key === "Escape") {
            event.stopPropagation();
            close();
            return;
        }
        if (event.key === "Tab" && modal.contains(event.target)) {
            const focusable = [...modal.querySelectorAll(
                'button:not([disabled]), select:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])'
            )].filter((element) => element.offsetParent !== null);
            if (!focusable.length) return;
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        }
    };
    const close = () => {
        document.removeEventListener("keydown", onKey, true);
        modal.dispatchEvent(new Event("mmrp-modal-close"));
        overlay.remove();
        if (previouslyFocused && typeof previouslyFocused.focus === "function") {
            previouslyFocused.focus();
        }
    };
    closeButton.onclick = close;
    overlay.onclick = (event) => {
        if (event.target === overlay) close();
    };
    // Keep graph-level delete/queue shortcuts out while a select or checkbox owns focus,
    // without intercepting the event before it reaches that control.
    modal.addEventListener("keydown", (event) => event.stopPropagation());
    document.addEventListener("keydown", onKey, true);
    document.body.appendChild(overlay);
    requestAnimationFrame(() => closeButton.focus());
    return { overlay, modal, close };
}

export function appendFileOptions(select, files, current = "") {
    select.replaceChildren();
    const names = [...new Set([current, ...(files || [])].filter(Boolean))];
    for (const file of names) {
        const option = document.createElement("option");
        option.value = file;
        option.textContent = file;
        select.appendChild(option);
    }
    select.value = current || names[0] || "";
}
