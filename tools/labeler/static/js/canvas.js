export class Annotator {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.img = new Image();
        this.state = {
            bboxes: [],
            waypoints: [],
            command: 0,
            scale: 1,
            mode: 'bbox', // 'bbox' or 'waypoint'
            isDrawing: false,
            startPos: null,
            selectedIdx: -1
        };

        this._setupEvents();
    }

    loadImage(url) {
        return new Promise((resolve) => {
            this.img.onload = () => {
                this._resizeCanvas();
                this.render();
                resolve();
            };
            this.img.src = url;
        });
    }

    _resizeCanvas() {
        const container = this.canvas.parentElement;
        const ratio = this.img.width / this.img.height;
        let w = container.clientWidth;
        let h = w / ratio;

        if (h > container.clientHeight) {
            h = container.clientHeight;
            w = h * ratio;
        }

        this.canvas.width = w;
        this.canvas.height = h;
        this.state.scale = w / this.img.width;
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.img, 0, 0, this.canvas.width, this.canvas.height);

        // Draw BBoxes
        this.state.bboxes.forEach((box, i) => {
            const color = i === this.state.selectedIdx ? '#ff0000' : '#00ff00';
            this._drawBox(box, color);
        });

        // Draw Waypoints
        this._drawWaypoints();
    }

    _drawBox(box, color) {
        const { cx, cy, w, h } = box;
        const x = (cx - w/2) * this.canvas.width;
        const y = (cy - h/2) * this.canvas.height;
        const bw = w * this.canvas.width;
        const bh = h * this.canvas.height;

        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(x, y, bw, bh);
    }

    _drawWaypoints() {
        if (this.state.waypoints.length === 0) return;

        this.ctx.strokeStyle = '#3498db';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();

        this.state.waypoints.forEach((wp, i) => {
            const x = wp.x * this.canvas.width;
            const y = wp.y * this.canvas.height;
            if (i === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);

            this.ctx.fillStyle = '#3498db';
            this.ctx.fillRect(x-2, y-2, 4, 4);
        });
        this.ctx.stroke();
    }

    _setupEvents() {
        this.canvas.onmousedown = (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / this.canvas.width;
            const y = (e.clientY - rect.top) / this.canvas.height;

            if (this.state.mode === 'bbox') {
                this.state.isDrawing = true;
                this.state.startPos = { x, y };
            } else if (this.state.mode === 'waypoint') {
                this.state.waypoints.push({ x, y });
                this.render();
            }
        };

        this.canvas.onmousemove = (e) => {
            if (!this.state.isDrawing) return;
            // Preview logic ...
        };

        this.canvas.onmouseup = (e) => {
            if (!this.state.isDrawing) return;
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / this.canvas.width;
            const y = (e.clientY - rect.top) / this.canvas.height;

            const cx = (this.state.startPos.x + x) / 2;
            const cy = (this.state.startPos.y + y) / 2;
            const w = Math.abs(x - this.state.startPos.x);
            const h = Math.abs(y - this.state.startPos.y);

            this.state.bboxes.push({ cx, cy, w, h, category: 3 }); // Default Car
            this.state.isDrawing = false;
            this.render();
        };
    }
}
