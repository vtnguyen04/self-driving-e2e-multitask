import { API } from './api.js';
import { Annotator } from './canvas.js';

class App {
    constructor() {
        this.annotator = new Annotator('canvas');
        this.currentFilename = null;
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.loadFileList();
        console.log('App initialized');
    }

    async loadFileList() {
        const samples = await API.labels.list({ limit: 50 });
        const listEl = document.getElementById('file-list');
        listEl.innerHTML = samples.map(s => `
            <div class="file-item ${s.is_labeled ? 'labeled' : ''}" data-file="${s.filename}">
                ${s.filename}
            </div>
        `).join('');
    }

    setupEventListeners() {
        document.getElementById('file-list').onclick = async (e) => {
            const item = e.target.closest('.file-item');
            if (!item) return;

            const filename = item.dataset.file;
            this.currentFilename = filename;
            const data = await API.labels.get(filename);

            await this.annotator.loadImage(`/api/v1/labels/image/${filename}`);
            this.annotator.state.bboxes = data.bboxes || [];
            this.annotator.state.waypoints = data.waypoints || [];
            this.annotator.render();
        };

        document.getElementById('btn-save').onclick = async () => {
            if (!this.currentFilename) return;
            const data = {
                command: 0,
                bboxes: this.annotator.state.bboxes,
                waypoints: this.annotator.state.waypoints
            };
            await API.labels.save(this.currentFilename, data);
            alert('Saved!');
            this.loadFileList();
        };

        document.getElementById('btn-publish').onclick = async () => {
            const name = prompt('Version Name (e.g. v1):');
            if (name) {
                await API.versions.publish(name, 'Manual snapshot');
                alert('Version Published!');
            }
        };
    }
}

new App();
