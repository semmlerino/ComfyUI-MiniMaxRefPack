// The input-directory picker and the asset-first task plan dialog. What a click does
// to the references is decided in lib/references.js; this file draws the controls.

import { CAPS, KINDS } from "./layout.js";
import {
    TASK_ROLE_OPTIONS, assignTags, changeReferenceSource, cloneRefs, deriveTaskPlanState,
    ensureReplacementPrimaryRole, finishTaskPlan, hasDriverRole, initialTaskPlan, newReference,
    referencesFingerprint, setPrimaryVideo, soundtrackRoleUnavailable, toggleTaskRole,
} from "./references.js";
import { setWidget, widgetByName } from "./widgets.js";

// `host` is what lives in refpack.js: the shared modal and input-file helpers, the
// thumbnail and soundtrack-probe caches, and applyRefs, which commits new references.
export function createTaskPlanUi(host) {
    const {
        createPlannerModal, appendFileOptions, apiListFiles, thumbUrl, dropThumb,
        probeResults, applyRefs,
    } = host;

    async function openAddReferenceModal(node, kind) {
        if (node._mmrpRefs[`${kind}s`].length >= CAPS[kind]) return;
        const { modal, close } = createPlannerModal(`Add ${kind} reference`, "mmrp-add-modal");

        const hint = document.createElement("div");
        hint.className = "mmrp-modal-hint";
        hint.textContent = "Choose a file already in ComfyUI's input directory, or upload a new one.";
        modal.appendChild(hint);

        const select = document.createElement("select");
        select.className = "mmrp-source-select";
        select.disabled = true;
        const loading = document.createElement("option");
        loading.textContent = "Loading input directory…";
        select.appendChild(loading);
        modal.appendChild(select);

        const status = document.createElement("div");
        status.className = "mmrp-plan-status";
        modal.appendChild(status);

        const footer = document.createElement("div");
        footer.className = "mmrp-modal-footer";
        const upload = document.createElement("button");
        upload.className = "mmrp-btn";
        upload.textContent = "Upload new…";
        upload.onclick = () => {
            close();
            node._mmrpBody.fileInputs[kind].click();
        };
        const cancel = document.createElement("button");
        cancel.className = "mmrp-btn";
        cancel.textContent = "Cancel";
        cancel.onclick = close;
        const add = document.createElement("button");
        add.className = "mmrp-btn mmrp-btn-primary";
        add.textContent = "Add selected";
        add.disabled = true;
        add.onclick = () => {
            const file = select.value;
            if (!file) return;
            // Re-check after the async listing: a simultaneous drop/upload may have filled
            // the last slot while this modal was open.
            const next = cloneRefs(node._mmrpRefs);
            const target = next[`${kind}s`];
            if (target.length >= CAPS[kind]) {
                status.textContent = `The ${kind} references are already at their ${CAPS[kind]}-asset limit.`;
                add.disabled = true;
                return;
            }
            target.push(newReference(kind, file));
            close();
            applyRefs(node, next);
        };
        footer.appendChild(upload);
        footer.appendChild(cancel);
        footer.appendChild(add);
        modal.appendChild(footer);

        try {
            const files = await apiListFiles(kind);
            appendFileOptions(select, files);
            select.disabled = files.length === 0;
            add.disabled = files.length === 0;
            status.textContent = files.length
                ? `${files.length} ${kind} file${files.length === 1 ? "" : "s"} available.`
                : `No ${kind} files are in the input directory yet.`;
        } catch (error) {
            status.textContent = `Could not list the input directory: ${error.message}`;
        }
    }

    async function openTaskPlanModal(node) {
        const openedFingerprint = referencesFingerprint(node._mmrpRefs);
        const working = cloneRefs(node._mmrpRefs);
        const legacyJobType = String(widgetByName(node, "job_type")?.value || "auto");
        if (!working.taskPlan) working.taskPlan = initialTaskPlan(legacyJobType);

        const { modal, close } = createPlannerModal("Task plan", "mmrp-task-modal");
        const hint = document.createElement("div");
        hint.className = "mmrp-modal-hint";
        hint.textContent = node._mmrpRefs.taskPlan
            ? "Choose what each reference contributes. Roles determine the exact MiniMax task prefix. Changing a source clears that asset's crop/trim edits; a new video starts with soundtrack detection on."
            : `This workflow currently uses legacy job_type=${legacyJobType}. Applying this dialog moves it to the asset-first plan; changing a source clears that asset's crop/trim edits, and a new video starts with soundtrack detection on.`;
        modal.appendChild(hint);

        const controls = document.createElement("div");
        controls.className = "mmrp-plan-controls";
        const modeLabel = document.createElement("label");
        modeLabel.textContent = "Planning";
        const mode = document.createElement("select");
        mode.innerHTML = '<option value="auto">Infer roles</option><option value="explicit">Set roles explicitly</option>';
        mode.value = working.taskPlan.mode;
        modeLabel.appendChild(mode);
        controls.appendChild(modeLabel);

        const specializationLabel = document.createElement("label");
        specializationLabel.textContent = "Replacement specialization";
        const specialization = document.createElement("select");
        specialization.innerHTML = [
            ["none", "None / general"],
            ["character_replacement", "Character replacement"],
            ["object_replacement", "Object replacement"],
        ].map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
        specialization.value = working.taskPlan.specialization || "none";
        specializationLabel.appendChild(specialization);
        controls.appendChild(specializationLabel);
        modal.appendChild(controls);

        const status = document.createElement("div");
        status.className = "mmrp-plan-status";
        modal.appendChild(status);

        const assets = document.createElement("div");
        assets.className = "mmrp-plan-assets";
        modal.appendChild(assets);

        const footer = document.createElement("div");
        footer.className = "mmrp-modal-footer";
        const cancel = document.createElement("button");
        cancel.className = "mmrp-btn";
        cancel.textContent = "Cancel";
        cancel.onclick = close;
        const save = document.createElement("button");
        save.className = "mmrp-btn mmrp-btn-primary";
        save.textContent = "Apply plan";
        footer.appendChild(cancel);
        footer.appendChild(save);
        modal.appendChild(footer);

        const fileLists = { image: [], video: [], audio: [] };
        const listingErrors = [];
        await Promise.all(KINDS.map(async (kind) => {
            try {
                fileLists[kind] = await apiListFiles(kind);
            } catch (error) {
                listingErrors.push(`${kind}: ${error.message}`);
            }
        }));

        // Every role checkbox and primary-video checkbox, each with its reference, so a
        // change anywhere re-derives all of them from the working copy.
        const roleInputs = [];     // { input, reference, roleId, kind }
        const primaryInputs = [];  // { input, reference }

        const update = () => {
            working.taskPlan.mode = mode.value;
            working.taskPlan.specialization = mode.value === "explicit" ? specialization.value : "none";
            ensureReplacementPrimaryRole(working);
            const explicit = mode.value === "explicit";
            specialization.disabled = !explicit;
            for (const { input, reference, roleId, kind } of roleInputs) {
                input.checked = (reference.roles || []).includes(roleId);
                input.disabled = !explicit || (kind === "video"
                    && soundtrackRoleUnavailable(reference, roleId, probeResults.get(reference.file)));
            }
            for (const { input, reference } of primaryInputs) {
                input.checked = reference.primary === true;
                input.disabled = !explicit || !hasDriverRole(reference);
            }
            const stale = referencesFingerprint(node._mmrpRefs) !== openedFingerprint;
            const state = deriveTaskPlanState(working, true);
            status.textContent = stale
                ? "References changed while this planner was open. Close it and reopen to keep the new assets."
                : state.error || (
                state.mode === "explicit"
                    ? `Derived prefix: ${state.prefix}`
                    : "The VLM will infer relationships; the existing classifier remains available for replacements."
            );
            status.classList.toggle("mmrp-plan-status-error", stale || !!state.error);
            status.title = listingErrors.length
                ? `Input listing warning — ${listingErrors.join("; ")}`
                : "";
            save.disabled = stale || !!state.error;
        };

        const assetRows = [];
        const refreshTagLabels = () => {
            const tagged = assignTags(working);
            for (const item of assetRows) {
                const taggedReference = tagged[`${item.kind}s`][item.index];
                item.tag.textContent = taggedReference.audioTag
                    ? `${taggedReference.tag} · soundtrack ${taggedReference.audioTag}`
                    : taggedReference.tag;
            }
        };

        for (const kind of KINDS) {
            working[`${kind}s`].forEach((reference, index) => {
                const row = document.createElement("section");
                row.className = "mmrp-plan-asset";

                const assetHead = document.createElement("div");
                assetHead.className = "mmrp-plan-asset-head";
                const identity = document.createElement("div");
                identity.className = "mmrp-plan-asset-identity";
                let thumbnail = null;
                if (kind === "image" || kind === "video") {
                    const thumbnailFrame = document.createElement("div");
                    thumbnailFrame.className = "mmrp-plan-thumbnail";
                    const thumbnailFallback = document.createElement("span");
                    thumbnailFallback.textContent = "No preview";
                    thumbnail = document.createElement("img");
                    thumbnail.className = "mmrp-plan-thumbnail-image";
                    thumbnail.loading = "lazy";
                    thumbnail.decoding = "async";
                    thumbnail.onload = () => thumbnailFrame.classList.add("mmrp-plan-thumbnail-ready");
                    thumbnail.onerror = () => thumbnailFrame.classList.remove("mmrp-plan-thumbnail-ready");
                    thumbnailFrame.appendChild(thumbnailFallback);
                    thumbnailFrame.appendChild(thumbnail);
                    identity.appendChild(thumbnailFrame);
                }
                const tag = document.createElement("strong");
                const kindLabel = document.createElement("span");
                kindLabel.textContent = kind;
                identity.appendChild(tag);
                assetHead.appendChild(identity);
                assetHead.appendChild(kindLabel);
                row.appendChild(assetHead);
                assetRows.push({ kind, index, tag });

                const refreshThumbnail = () => {
                    if (!thumbnail) return;
                    thumbnail.alt = `${kind === "image" ? "Picture" : "Video"} preview: ${reference.file}`;
                    thumbnail.src = thumbUrl(reference.file, reference);
                };
                refreshThumbnail();

                const sourceLabel = document.createElement("label");
                sourceLabel.className = "mmrp-source-row";
                sourceLabel.textContent = "Source";
                const source = document.createElement("select");
                source.className = "mmrp-source-select";
                appendFileOptions(source, fileLists[kind], reference.file);
                source.onchange = () => {
                    const oldFile = reference.file;
                    if (!source.value || source.value === oldFile) return;
                    changeReferenceSource(reference, kind, source.value, probeResults.get(source.value));
                    dropThumb(oldFile);
                    refreshThumbnail();
                    refreshTagLabels();
                    update();
                };
                sourceLabel.appendChild(source);
                row.appendChild(sourceLabel);

                if (kind === "video") {
                    const primaryBlock = document.createElement("div");
                    primaryBlock.className = "mmrp-primary-block";
                    const primaryLabel = document.createElement("label");
                    primaryLabel.className = "mmrp-primary-row";
                    const primaryControl = document.createElement("input");
                    primaryControl.type = "checkbox";
                    primaryControl.dataset.primaryControl = "true";
                    primaryControl.onchange = () => {
                        setPrimaryVideo(working, reference, primaryControl.checked);
                        update();
                    };
                    primaryInputs.push({ input: primaryControl, reference });
                    primaryLabel.appendChild(primaryControl);
                    primaryLabel.appendChild(document.createTextNode("Primary video"));
                    primaryBlock.appendChild(primaryLabel);
                    const primaryHelp = document.createElement("span");
                    primaryHelp.className = "mmrp-primary-help";
                    primaryHelp.textContent = "Choose the editing or continuation source that drives the result. Replacement plans use the editing source.";
                    primaryBlock.appendChild(primaryHelp);
                    row.appendChild(primaryBlock);
                }

                const roles = document.createElement("div");
                roles.className = "mmrp-role-grid";
                for (const [roleId, roleLabel] of TASK_ROLE_OPTIONS[kind]) {
                    const label = document.createElement("label");
                    const checkbox = document.createElement("input");
                    checkbox.type = "checkbox";
                    checkbox.value = roleId;
                    checkbox.dataset.roleControl = "true";  // the UI smoke selects by these two
                    checkbox.onchange = () => {
                        toggleTaskRole(working, kind, reference, roleId, checkbox.checked);
                        update();
                    };
                    roleInputs.push({ input: checkbox, reference, roleId, kind });
                    label.appendChild(checkbox);
                    label.appendChild(document.createTextNode(roleLabel));
                    roles.appendChild(label);
                }
                row.appendChild(roles);
                assets.appendChild(row);
            });
        }
        refreshTagLabels();
        if (!assets.children.length) {
            const empty = document.createElement("div");
            empty.className = "mmrp-plan-empty";
            empty.textContent = "Add a reference first; the node-wide drop zone and upload buttons still work.";
            assets.appendChild(empty);
        }

        mode.onchange = update;
        specialization.onchange = update;
        save.onclick = () => {
            if (referencesFingerprint(node._mmrpRefs) !== openedFingerprint) {
                update();
                return;
            }
            if (finishTaskPlan(working, node._mmrpRefs.control).error) return;
            setWidget(node, "job_type", "auto");
            close();
            applyRefs(node, working);
        };
        update();
    }

    return { openAddReferenceModal, openTaskPlanModal };
}
