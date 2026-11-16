// Ask before opening a UP scan when no companion was selected.
export function createUpScanPrompt(onInvalid) {
  let active = null;
  function retire() { if (active) active.close(); }
  function choose(file) {
    retire();
    return new Promise((resolve) => {
      const dialog = document.createElement("dialog");
      dialog.className = "up-scan-dialog";
      const title = document.createElement("h2");
      title.id = "up-scan-title";
      dialog.setAttribute("aria-labelledby", title.id);
      title.textContent = `Add an ANG file for ${file.name}?`;
      const detail = document.createElement("p");
      detail.id = "up-scan-detail";
      detail.textContent = "Optional. If available, add the matching ANG file for the scan layout and image quality map.";
      dialog.setAttribute("aria-describedby", detail.id);
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".ang";
      input.hidden = true;
      const error = document.createElement("div");
      error.className = "up-scan-error";
      error.setAttribute("role", "alert");
      const buttons = document.createElement("div");
      const button = (label, action) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.addEventListener("click", action);
        buttons.append(b);
      };
      let result = null;
      button("Choose ANG…", () => input.click());
      button("Open without ANG", () => { result = []; dialog.close(); });
      button("Cancel", () => dialog.close());
      input.addEventListener("change", () => {
        if (!input.files.length) return;
        const selected = input.files[0];
        input.value = "";
        if (!/\.ang$/i.test(selected.name)) { error.textContent = onInvalid(selected); return; }
        result = [selected];
        dialog.close();
      });
      dialog.addEventListener("close", () => {
        if (active === dialog) active = null;
        dialog.remove();
        resolve(result);
      }, { once: true });
      dialog.append(title, detail, input, error, buttons);
      document.body.append(dialog);
      active = dialog;
      dialog.showModal();
    });
  }
  return { choose, retire };
}
