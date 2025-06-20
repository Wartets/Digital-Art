const canvas = document.getElementById("fractal");
const ctx = canvas.getContext("2d");
let w = canvas.width = window.innerWidth;
let h = canvas.height = window.innerHeight;

const maxFramesStored = Math.floor(Math.random() * 5) + 3;
const previousFrames = [];

function makeSeededRandom(seed) {
	let s = seed % 2147483647;
	if (s <= 0) s += 2147483646;
	return function() {
		s = s * 16807 % 2147483647;
		return (s - 1) / 2147483646;
	};
}

const seed = Math.random();
const randSeed = makeSeededRandom(seed);
const rand = (a, b) => randSeed() * (b - a) + a;

// --- CONSTANTES DE CONTRÔLE ---
const A = 10 * randSeed() + 6;			// nombre de formes
const B = rand(0.15, 0.3);				// min position (% w/h)
const C = rand(0.5, 1);					// max position (% w/h)
const D = rand(30, 40);					// rayon min
const E = rand(150, 250);				// rayon max
const F = rand(0.5, 1) / 10;			// déformation min
const G = randSeed() + 0.3;				// déformation max
const H = rand(0.2, 1) * 3;				// nb de côtés min
const I = rand(0, 0.6);					// nb de côtés max
const J = rand(0.003, 0.05);			// influence du champ
const K = rand(10, 1200) / 10;				// intensité de couleur (multiplicateur)
const L = rand(0, 360);					// gamme de teintes
const QUALITY = 50;						// facteur de qualité (1 = full, 2 = demi-rendu, etc.)
const QUALITY2 = 2; 					// Qualité globale (1 = plein, 2 = demi-résolution, etc.)

const globalDeformTypes = ["sin", "cos", "tan", "waves", "none"];
const globalDeformType = globalDeformTypes[Math.floor(rand(0, globalDeformTypes.length))];

function makeShapes(n) {
	const types = [
		"circle", "ellipse", "polygon", "star", "flower",
		"heart", "rect", "cross", "gear", "wave", "ring"
	];
	const deformTypes = ["sine", "ripple", "noise"];
	const shapes = [];

	for (let i = 0; i < n; i++) {
		const type = types[Math.floor(rand(0, types.length))];
		const deform = rand(F, G);
		const deformType = deformTypes[Math.floor(rand(0, deformTypes.length))];

		shapes.push({
			index: i,
			x: rand(B, C) * w,
			y: rand(B, C) * h,
			r: rand(D, E),
			type: type,
			rot: rand(0, Math.PI * 2),
			deform: deform,
			points: Math.floor(rand(H, I)),
			deformType: deformType,
			applyDeform: randSeed() > 0.5,
			reach: rand(1.2, 2.5) * E,
			mod1: rand(0.5, 1.5 * G * 5),
			mod2: randSeed() > 0.5 ? deform * rand(0.3, 2 * G) : 0,
			rot2: rand(0, 2 * Math.PI),
			mix: Math.random() > 0.8
		});
	}

	return shapes;
}

let blurActive = false;

window.addEventListener("keydown", (e) => {
  // e.key contient la touche en string
  if (
    (e.key.length === 1 && e.key.match(/[a-zA-Z]/)) || // lettre
    e.key === " " // espace
  ) {
    blurActive = true;
  }
});

window.addEventListener("keyup", (e) => {
  if (
    (e.key.length === 1 && e.key.match(/[a-zA-Z]/)) ||
    e.key === " "
  ) {
    blurActive = false;
  }
});


function isoDistance(x, y, shape, time) {
	const dx = x - shape.x;
	const dy = y - shape.y;
	const dist = Math.sqrt(dx * dx + dy * dy);
	const angle = Math.atan2(dy, dx) - shape.rot;
	const a2 = angle + (shape.rot2 || 0);
	const dNorm = dist / shape.r;

	const modulate = (base) => {
		let v = base;
		switch (shape.deformType) {
			case "sine":
				if (shape.mod1) v *= (1 + shape.deform * Math.sin(shape.mod1 * a2));
				if (shape.mod2) v *= (1 + shape.deform * Math.sin(shape.mod2 * dNorm + a2));
				break;

			case "droplet":
				v *= (1 + shape.deform * Math.exp(-dNorm * 4) * Math.sin(time * 10 + dNorm * 20));
				break;

			case "ripple":
				v *= (1 + shape.deform * Math.sin(dist * 0.1 - time * 5));
				break;

			case "noise":
				const n = Math.sin(12.9898 * dx + 78.233 * dy + 43758.5453) * 43758.5453;
				const noise = (n - Math.floor(n)) * 2 - 1;
				v *= (1 + shape.deform * noise);
				break;
		}
		return v;
	};

	let pulseFactor = 1;
	for (let p of pulseCenters) {
		if (p.shapeIndex === undefined) continue;
		if (p.shapeIndex !== shape.index) continue;
		const pdx = x - p.x;
		const pdy = y - p.y;
		const pdist = Math.sqrt(pdx * pdx + pdy * pdy);
		if (pdist < p.radius) {
			const normDist = pdist / p.radius;
			const pulse = 1 + p.intensity * Math.sin(time * p.frequency * 2 * Math.PI + p.phase) * (1 - normDist);
			pulseFactor *= pulse;
		}
	}

	switch (shape.type) {
		case "circle":
			return (dist - shape.r) * pulseFactor;

		case "ring": {
			const inner = shape.r * (1 - shape.deform);
			const outer = shape.r;
			return (Math.abs(dist - (inner + outer) / 2) - (outer - inner) / 2) * pulseFactor;
		}

		case "ellipse":
			return ((Math.pow(dx / shape.r, 2) + Math.pow(dy / (shape.r * shape.deform), 2)) - 1) * pulseFactor;

		case "polygon":
		case "star": {
			const freq = shape.points * (shape.type === "star" ? 2 : 1);
			const rMod = modulate(shape.r * (1 + shape.deform * Math.sin(freq * a2)));
			return (dist - rMod) * pulseFactor;
		}

		case "flower": {
			const petals = shape.points * 2;
			const rMod = modulate(shape.r * (0.5 + 0.5 * Math.sin(petals * a2)));
			return (dist - rMod) * pulseFactor;
		}

		case "heart": {
			const nx = dx / shape.r;
			const ny = dy / shape.r;
			const a = nx * nx + ny * ny - 1;
			return (Math.pow(a, 3) - nx * nx * ny * ny * ny) * pulseFactor;
		}

		case "rect": {
			const hw = shape.r;
			const hh = shape.r * shape.deform;
			const qx = Math.abs(dx) - hw;
			const qy = Math.abs(dy) - hh;
			const ex = Math.max(qx, 0);
			const ey = Math.max(qy, 0);
			return (Math.sqrt(ex * ex + ey * ey) + Math.min(Math.max(qx, qy), 0)) * pulseFactor;
		}

		case "cross": {
			const size = shape.r;
			const thickness = shape.r * shape.deform;
			const d1 = Math.max(Math.abs(dx) - thickness, 0);
			const d2 = Math.max(Math.abs(dy) - thickness, 0);
			return (Math.min(Math.max(Math.abs(dx) - size, Math.abs(dy)),
											Math.max(Math.abs(dy) - size, Math.abs(dx))) - thickness / 2) * pulseFactor;
		}

		case "gear": {
			const teeth = shape.points * 2;
			const rMod = modulate(shape.r * (1 + shape.deform * Math.sin(teeth * a2)));
			return (dist - rMod) * pulseFactor;
		}

		case "wave": {
			const freq = shape.points;
			const rMod = modulate(shape.r + shape.r * shape.deform * Math.sin(freq * dNorm + a2));
			return (dist - rMod) * pulseFactor;
		}
	}

	let base = dist - shape.r;
	if (shape.mix) {
		base *= (1 + shape.deform * Math.sin(shape.points * a2) * Math.cos(dNorm));
	}
	return base * pulseFactor;
}

const pulseCount = Math.floor(rand(3, 5));
const pulseCenters = [];
const pulseMaxRadius = E * 1.2 + K / 2;

for (let i = 0; i < pulseCount; i++) {
	const shapeIndex = Math.floor(rand(0, A));
	pulseCenters.push({
		x: rand(B, C) * w,
		y: rand(B, C) * h,
		radius: rand(D * 0.5, pulseMaxRadius),
		shapeIndex: shapeIndex,
		intensity: rand(0.1, 0.5),
		frequency: rand(0.002, 0.008) * 1000,
		phase: rand(0, Math.PI * 2)
	});
}

function drawArt(time) {
	const w2 = Math.floor(w / QUALITY2);
	const h2 = Math.floor(h / QUALITY2);

	const tempCanvas = document.createElement("canvas");
	tempCanvas.width = w2;
	tempCanvas.height = h2;
	const tempCtx = tempCanvas.getContext("2d");
	const img = tempCtx.createImageData(w2, h2);
	const data = img.data;
	const shapes = makeShapes(A);
	const colorOffset = L;

	for (let y = 0; y < h2; y += QUALITY) {
		for (let x = 0; x < w2; x += QUALITY) {
			let d = 0;
			const px = x * QUALITY2;
			const py = y * QUALITY2;

			for (let s of shapes) {
				const dx = px - s.x;
				const dy = py - s.y;
				const dist2 = dx * dx + dy * dy;
				if (dist2 < s.reach * s.reach) {
					const iso = isoDistance(px, py, s, time);
					d += Math.exp(-iso * iso * J);
				}
			}

			const i = (y * w2 + x) * 4;
			const hue = (colorOffset + d * (K + time * 10)) % 360;
			const light = Math.min(100, d * 50);
			const [r, g, b] = hslToRgb(hue + x / (L + 0.1), 1, light / 100);

			for (let dy = 0; dy < QUALITY; dy++) {
				for (let dx = 0; dx < QUALITY; dx++) {
					const j = ((y + dy) * w2 + (x + dx)) * 4;
					if (j < data.length) {
						data[j] = r;
						data[j + 1] = g;
						data[j + 2] = b;
						data[j + 3] = 255;
					}
				}
			}
		}
	}

	tempCtx.putImageData(img, 0, 0);

	const original = tempCtx.getImageData(0, 0, w2, h2);
	const output = tempCtx.createImageData(w2, h2);
	const pixels = original.data;
	const out = output.data;

	const strength = G * E + A * 20;
	const freq = K;
	const chaos = F * 2000;

	for (let y = 0; y < h2; y++) {
		for (let x = 0; x < w2; x++) {
			const i = (y * w2 + x) * 4;

			let dx = x + A * Math.sin(time);
			let dy = y + B * Math.sin(time);

			switch (globalDeformType) {
				case "sin":
					dx += chaos * Math.sin(y / freq + L) + strength * Math.sin(y / 70 + A) * Math.sin(time / E);
					dy += chaos * Math.cos(x / freq + K) + strength * Math.cos(x / 90 + G * 10) * Math.sin(time / E);
					break;

				case "cos":
					dx += chaos * Math.cos(y / freq + L) + strength * Math.sin(y / 50 + A) * Math.sin(time / E);
					dy += chaos * Math.sin(x / freq + K * Math.sin(time / E)) + strength * Math.cos(x / 60 + G * 10);
					break;

				case "tan":
					dx += chaos * Math.tan(y / (L+0.1) + L * Math.sin(time / E)) % 20 * Math.sin(time / E);
					dy += chaos * Math.tan(x / K + K) % 20 * Math.sin(time / E);
					break;

				case "waves":
					dx += D * Math.sin(y / 25 + time) * Math.sin(time / E);
					dy += D * Math.sin(x / 25 + time) * Math.sin(time / E);
					break;

				case "none":
					// Pas de déformation
					break;
			}

			dx = Math.floor(dx);
			dy = Math.floor(dy);

			if (dx >= 0 && dx < w2 && dy >= 0 && dy < h2) {
				const j = (dy * w2 + dx) * 4;
				out[i] = pixels[j];
				out[i + 1] = pixels[j + 1];
				out[i + 2] = pixels[j + 2];
				out[i + 3] = 255;
			}
		}
	}

	tempCtx.putImageData(output, 0, 0);

	const blended = tempCtx.getImageData(0, 0, w2, h2);
	previousFrames.push(blended);
	if (previousFrames.length > maxFramesStored) previousFrames.shift();

	const finalImage = tempCtx.createImageData(w2, h2);
	const finalData = finalImage.data;
	const blendCount = previousFrames.length;

	const alphas = [];
	let alphaSum = 0;
	for (let k = 0; k < blendCount; k++) {
		const alpha = (k + 1) / blendCount;	
		alphas.push(alpha);
		alphaSum += alpha;
	}

	for (let i = 0; i < finalData.length; i += 4) {
		let r = 0, g = 0, b = 0;
		for (let k = 0; k < blendCount; k++) {
		const frameData = previousFrames[k].data;
		const a = alphas[k] / alphaSum;
		r += frameData[i] * a;
		g += frameData[i + 1] * a;
		b += frameData[i + 2] * a;
		}
		finalData[i] = r;
		finalData[i + 1] = g;
		finalData[i + 2] = b;
		finalData[i + 3] = 255;
	}

	tempCtx.putImageData(finalImage, 0, 0);

	ctx.clearRect(0, 0, w, h);

	if (blurActive) {
	  ctx.filter = `blur(${E / 5}px)`;
	  ctx.drawImage(tempCanvas, 0, 0, w, h);
	  ctx.filter = 'none';
	} else {
	  ctx.drawImage(tempCanvas, 0, 0, w, h);
	}
}



function hslToRgb(h, s, l) {
	h /= 360;
	let r, g, b;
	if (s === 0) {
		r = g = b = l;
	} else {
		const hue2rgb = (p, q, t) => {
			if (t < 0) t += 1;
			if (t > 1) t -= 1;
			if (t < 1 / 6) return p + (q - p) * 6 * t;
			if (t < 1 / 2) return q;
			if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
			return p;
		};
		const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
		const p = 2 * l - q;
		r = hue2rgb(p, q, h + 1 / 3);
		g = hue2rgb(p, q, h);
		b = hue2rgb(p, q, h - 1 / 3);
	}
	return [Math.floor(r * 255), Math.floor(g * 255), Math.floor(b * 255)];
}

window.addEventListener("resize", () => {
	w = canvas.width = window.innerWidth;
	h = canvas.height = window.innerHeight;
});

let start = null;
function animate(timestamp) {
	if (!start) start = timestamp;
	const elapsed = (timestamp - start) / 1000 * 100 * maxFramesStored;
	drawArt(elapsed);
	requestAnimationFrame(animate);
}
requestAnimationFrame(animate);