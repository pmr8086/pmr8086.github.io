/** 0 - MATH **/

const
	pi = Math.PI,
	abs = Math.abs,
	sin = Math.sin,
	cos = Math.cos,
	sinh = Math.sin,
	cosh = Math.cosh,
	tan = Math.tan,
	acos = Math.acos,
	asin = Math.asin,
	atan = Math.atan,
	atan2 = Math.atan2,
	log = Math.log,
	exp = Math.exp,
	sqrt = Math.sqrt,
	floor = Math.floor,
	Delta = (phi, k) => sqrt(1 - (k*sin(phi))**2),

	ex = [1, 0, 0],
	ey = [0, 1, 0],
	ez = [0, 0, 1];

function am(u, k) {
	if (u < 0) return -am(-u, k);
	var K = F(pi/2, k);
	var phii = floor(u / (2*K));
	if (phii > 0)
		return pi*phii + am(u % (2*K), k);

	var ks = [];
	while (1.0 - k > 1e-12) {
		ks.push(k);
		k = 2*sqrt(k)/(1+k);
	}
	var uN = u;
	for (k of ks)
		uN *= (1+k)/2;
	var phi = 2 * atan(exp(uN)) - pi/2;
	for (var n = ks.length - 1; n >= 0; n--)
		phi = atan2(sin(2*phi),ks[n]+cos(2*phi));
	return phi;

}

function F(phi, k) {
	var ks = [];
	while (1.0 - k > 1e-12) {
		ks.push(k);
		phi = (phi + asin(k*sin(phi)))/2;
		k = 2*sqrt(k)/(1+k);
	}
	var F = log(tan(phi/2+pi/4));
	for (k of ks)
		F *= 2/(1+k);
	return F;
}

/** 1 - WEBGL INITIALIZATION **/

const
	canvas = document.getElementById("canvas"),
	gl = canvas.getContext("webgl");

gl.enable(gl.DEPTH_TEST);
gl.clearColor(.3, .3, .6, 1.);

/** 2 - MATRIX **/

function Transform(matrix) {
	this.matrix = (matrix != undefined) ? matrix :
	[
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	];
}

Transform.prototype.transl = function(x, y, z) {
	this.lmulby(transl(x,y,z));
}

Transform.prototype.rot = function(degrees, axis) {
	this.lmulby(rot(degrees, axis));
}

Transform.prototype.r0t = function(axis) {
	this.lmulby(r0t(axis));
}

Transform.prototype.lmulby = function (A) {
	this.matrix = mul(A, this).matrix;
}

Transform.prototype.rmulby = function (B) {
	this.matrix = mul(this, B).matrix;
}

Transform.prototype.log = function (num_digits = 4) {
	var str = "%c";
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++){
			let el = this.matrix[4*col + row];
			if (el >= 0) str += " ";
			str += el.toFixed(num_digits) + " ";
		}
		str += "\n";
	}
	console.log(str, "font-family: monospace");
}

function mul(A, B) {
	// result = A * B
	var result = new Transform();
	for (let row = 0; row < 4; row++) {
		for (let col = 0; col < 4; col++) {
			let sum = 0;
			for (let k = 0; k < 4; k++)
				sum += A.matrix[4*k + row] * B.matrix[4*col + k]
			result.matrix[4*col + row] = sum;
		}
	}
	return result;
}

function vmul(A, v) {
	var result = [0, 0, 0];
	for (let row = 0; row < 4; row++) {
		let sum = 0;
		for (let k = 0; k < 4; k++)
			sum += A.matrix[4*k + row] * v[k];
		result[row] = sum;
	}
	return result;
}

function transl(x, y, z) {
	return new Transform([
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		x, y, z, 1
	]);
}

function rot(degrees, axis) {
	var
		m = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]),
		n = isNaN(m) ? { x: 0, y: 0, z: 0 } : {
			x: axis[0]/m,
			y: axis[1]/m,
			z: axis[2]/m
		},
		Phi = degrees * pi / 180,
		c = cos(Phi),
		d = 1 - c,
		s = sin(Phi);

	return new Transform([
		d*n.x*n.x  +  c, d*n.x*n.y+s*n.z, d*n.x*n.z-s*n.y, 0,
		d*n.y*n.x-s*n.z, d*n.y*n.y  +  c, d*n.y*n.z+s*n.x, 0,
		d*n.z*n.x+s*n.y, d*n.z*n.y-s*n.x, d*n.z*n.z  +  c, 0,
		0              , 0              , 0              , 1
	]);
}

function r0t(axis) {
	// this is like "rot",
	// except now the amount of degrees is
	// inferred from the magnitude
	// of the axis

	var
		m = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]),
		n = isNaN(m) ? { x: 0, y: 0, z: 0 } : {
			x: axis[0]/m,
			y: axis[1]/m,
			z: axis[2]/m
		},
		Phi = m,
		c = cos(Phi),
		d = 1 - c,
		s = sin(Phi);

	return new Transform([
		d*n.x*n.x  +  c, d*n.x*n.y+s*n.z, d*n.x*n.z-s*n.y, 0,
		d*n.y*n.x-s*n.z, d*n.y*n.y  +  c, d*n.y*n.z+s*n.x, 0,
		d*n.z*n.x+s*n.y, d*n.z*n.y-s*n.x, d*n.z*n.z  +  c, 0,
		0              , 0              , 0              , 1
	]);
}

function rot2(from, to) {
	var frommag = sqrt(from[0]*from[0] + from[1]*from[1] + from[2]*from[2]),
		tomag = sqrt(to[0]*to[0] + to[1]*to[1] + to[2]*to[2]);

	var
		v = [
			(from[1]*to[2]-to[1]*from[2])/tomag/frommag,
			(from[2]*to[0]-to[2]*from[0])/tomag/frommag,
			(from[0]*to[1]-to[0]*from[1])/tomag/frommag
		],
		c = (from[0]*to[0] + from[1]*to[1] + from[2]*to[2])/tomag/frommag,
		vcross = new Transform([
			0, v[2], -v[1], 0,
			-v[2], 0, v[0], 0,
			v[1], -v[0], 0, 0,
			0, 0, 0, 1
			]),
		vcross2 = mul(vcross, vcross),
		id =  new Transform();
	return new Transform(id.matrix.map(
		(t,i) => t + (i!=15)*(vcross.matrix[i]+vcross2.matrix[i]/(1+c)
		)));
}

function diag(x, y, z) {
	return new Transform([
		x, 0, 0, 0,
		0, y, 0, 0,
		0, 0, z, 0,
		0, 0, 0, 1
	]);
}

/** 3 - MODEL **/

var models = [];

function Model(verts, normals, colors, matrix) {

	this.transform = new Transform(matrix);

	if (verts != undefined) {
		this.buffer_verts = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer_verts);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(verts), gl.STATIC_DRAW);

		this.no_verts = parseInt(verts.length / 3);

		this.buffer_normals = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer_normals);
		if (normals != undefined)
			gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);
		else
			gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(Array(verts.length).fill(0)), gl.STATIC_DRAW);

		
		this.buffer_colors = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer_colors);
		if (colors != undefined)
			gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
		else
			gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(Array(verts.length).fill(1)), gl.STATIC_DRAW);
	}

	models.push(this);
}

Model.prototype.draw = function () {
	bind_buffer2attrib(this.buffer_verts, gl.getAttribLocation(program, "vertex"));
	bind_buffer2attrib(this.buffer_normals, gl.getAttribLocation(program, "normal"));
	bind_buffer2attrib(this.buffer_colors, gl.getAttribLocation(program, "color"));
	send_matrix2gpu(this.transform.matrix, gl.getUniformLocation(program, "model"));
	gl.drawArrays(gl.TRIANGLES, 0, this.no_verts);
}

function add_ellipsoid(a, b, c, thetastep = 1e-1, phistep = 1e-1) {
	

	var
		verts = [],
		normals = [],
		colors = [],

		x = (theta, phi)  =>  a*sin(theta)*cos(phi),
		y = (theta, phi)  =>  b*sin(theta)*sin(phi),
		z = (theta, phi)  =>  c*cos(theta),
		nx= (theta, phi)  =>b*c*sin(theta)*cos(phi),
		ny= (theta, phi)  =>c*a*sin(theta)*sin(phi),
		nz= (theta, phi)  =>a*b*cos(theta),

		addresult = function() {
			var
				X = x(theta,phi),
				Y = y(theta,phi),
				Z = z(theta,phi);

			verts.push(X);
			verts.push(Y);
			verts.push(Z);
			normals.push(nx(theta,phi));
			normals.push(ny(theta,phi));
			normals.push(nz(theta,phi));

			if (/*X >= 0 ^ */Y >= 0/* ^ Z >= 0*/) {
				colors.push(1);
				colors.push(1);
				colors.push(1);
			} else {
				colors.push(.1);
				colors.push(.1);
				colors.push(.1);
			}

			// colors.push(1 * (X >= 0));
			// colors.push(1 * (Y >= 0));
			// colors.push(1 * (Z >= 0));

		};

		for (var theta = 0; theta < pi; theta += thetastep) {
			for (var phi = 0; phi < 2*pi;) {
				addresult();
				phi += phistep; addresult();
				theta += thetastep; addresult();
				theta -= 2*thetastep; addresult();
				theta += thetastep; phi += phistep; addresult();
				phi -= phistep; addresult();
			}
		}

		return new Model(verts, normals, colors);
}

function add_arrow(r1, h1, r2, h2, color = [1,1,1], zstep = 1e-2, phistep = 1e-1) {

	var
		verts = [],
		normals = [],
		colors = [],

	addresult1 = function() {
		verts.push(r1*cos(phi));
		verts.push(r1*sin(phi));
		verts.push(z);
		normals.push(cos(phi));
		normals.push(sin(phi));
		normals.push(0);

		colors.push(color[0]);
		colors.push(color[1]);
		colors.push(color[2]);
	},

	addresult2 = function() {
		let
			r = r2*(1-(z-h1)/h2),
			theta = atan(h2/r2);

		verts.push(r*cos(phi));
		verts.push(r*sin(phi));
		verts.push(z);
		normals.push(cos(phi)*sin(theta));
		normals.push(sin(phi)*sin(theta));
		normals.push(cos(theta));

		colors.push(color[0]);
		colors.push(color[1]);
		colors.push(color[2]);
	},

	z, phi;

	for (z = 0; z <= h1; z += zstep) {
		for (phi = 0; phi <= 2*pi; phi += phistep) {
			addresult1();
			phi += phistep; addresult1();
			z += zstep; addresult1(); addresult1();
			phi -= phistep; addresult1();
			z -= zstep; addresult1();
		}
	}

	for (; z < h1+h2; z += zstep) {
		for (phi = 0; phi <= 2*pi; phi += phistep) {
			addresult2();
			phi += phistep; addresult2();
			z += zstep; addresult2(); addresult2();
			phi -= phistep; addresult2();
			z -= zstep; addresult2();
		}
	}

	return new Model(verts, normals, colors);

}

/** 4 - SHADER **/

var vert_shader = gl.createShader(gl.VERTEX_SHADER);
gl.shaderSource(vert_shader,`
attribute vec3 vertex;
attribute vec3 normal;
attribute vec3 color;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
varying vec3 vNormal;
varying vec3 vCoords;
varying vec3 vColor;

void main() {
	vCoords = vertex;
	vNormal = (model * vec4(normal, 0.)).xyz;
	gl_Position = projection * view * model * vec4(vertex, 1.);
	vColor = color;
}
`);
gl.compileShader(vert_shader);

var frag_shader = gl.createShader(gl.FRAGMENT_SHADER);
gl.shaderSource(frag_shader,`
precision highp float;
varying vec3 vNormal;
varying vec3 vCoords;
varying vec3 vColor;
void main() {
	vec3 sunlight_direction = vec3(-1., -1., -1.);
	float lightness = -clamp(dot(normalize(vNormal), normalize(sunlight_direction)), -1., 0.);
	gl_FragColor = vec4(lightness * vColor, 1.);
}
`);
gl.compileShader(frag_shader);

var program = gl.createProgram();
gl.attachShader(program, vert_shader);
gl.attachShader(program, frag_shader);
gl.linkProgram(program);

gl.useProgram(program);

/** 5 - JS -> GPU **/

function bind_buffer2attrib(buffer, attrib) {
	gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
	gl.enableVertexAttribArray(attrib);
	gl.vertexAttribPointer(attrib, 3, gl.FLOAT, false, 0, 0);
}

function send_matrix2gpu(matrix, uniform) {
	gl.uniformMatrix4fv(uniform, false, new Float32Array(matrix));
}

/** 6 - CAMERA **/

function pers_proj(near, far, width, height) {
	return new Transform([
		2*near/width, 0, 0, 0,
		0, 2*near/height, 0, 0,
		0, 0, (far+near)/(near-far), -1,
		0, 0, 2*far*near/(near-far), 0
	]);
}

function ortho_proj(near, far, width, height) {
	return new Transform([
		2*near/width, 0, 0, 0,
		0, 2*near/height, 0, 0,
		0, 0, 2/(near-far), 0,
		0, 0, (far+near)/(near-far), 1
	]);
}

Camera = function(projection, view) {
	this.projection = projection;
	this.view = view;
	this.update();
}

Camera.prototype.update = function () {
	send_matrix2gpu(this.projection.matrix, gl.getUniformLocation(program, "projection"));
	send_matrix2gpu(this.view.matrix, gl.getUniformLocation(program, "view"));
}

/** 7 - DRAG **/

var X0, Y0, CV0, ALT, WAS_PLAYING;

canvas.addEventListener("dragstart", function (evt) {
	X0 = evt.x; Y0 = evt.y;
	CV0 = camera.view;
	ALT = evt.altKey;
	WAS_PLAYING = NOW_PLAYING;
	NOW_PLAYING = true;
	loop();
});
canvas.addEventListener("drag", function (evt) {
	var transl_factor = 250, angle_factor = 2;
	if (ALT) {
		camera.view = mul(transl(
				(evt.x - X0)/transl_factor,
				-(evt.y - Y0)/transl_factor,
				0.
			), CV0)
	} else {
		let dx = CV0.matrix[12],
			dy = CV0.matrix[13],
			dz = CV0.matrix[14],
			CV = new Transform(CV0.matrix),
			DX = evt.x - X0,
			DY = evt.y - Y0;
		CV.transl(-dx,-dy,-dz);
		CV = mul(rot(sqrt(DX*DX + DY*DY)/angle_factor, [DY, DX, 0.]), CV);
		CV.transl(dx, dy, dz);
		camera.view = CV;
		
	}
	camera.update();
});
canvas.addEventListener("dragend", function () {
	NOW_PLAYING = WAS_PLAYING;
})

canvas.addEventListener("touchstart", function (evt) {
	X0 = evt.touches[0].pageX;
	Y0 = evt.touches[0].pageY;
	CV0 = camera.view;
	WAS_PLAYING = NOW_PLAYING;
	NOW_PLAYING = true;
	loop();
});
canvas.addEventListener("touchmove", function (evt) {
	evt.preventDefault();
	var angle_factor = 2;
	let dx = CV0.matrix[12],
		dy = CV0.matrix[13],
		dz = CV0.matrix[14],
		CV = new Transform(CV0.matrix),
		DX = (evt.touches[0].pageX - X0),
		DY = (evt.touches[0].pageY - Y0);
	CV.transl(-dx,-dy,-dz);
	CV = mul(rot(sqrt(DX*DX + DY*DY)/angle_factor, [DY, DX, 0.]), CV);
	CV.transl(dx, dy, dz);
	camera.view = CV;
	camera.update();
});
canvas.addEventListener("touchend", function () {
	NOW_PLAYING = WAS_PLAYING;
});

/** 9 - MOTION **/

function mxn_params(R, W0) {

	I = [
		R[1]*R[1] + R[2]*R[2],
		R[2]*R[2] + R[0]*R[0],
		R[0]*R[0] + R[1]*R[1]
	];
	
	d = sqrt((I[0]*W0[0])**2+(I[1]*W0[1])**2+(I[2]*W0[2])**2);

	c = I[0]*W0[0]**2+I[1]*W0[1]**2+I[2]*W0[2]**2;

	var s = Array(3);
	if ((s[0] = I[0]==I[1])+(s[1] = I[1]==I[2])+(s[2] = I[2]==I[1]) > 0) {
		let
			i = s.indexOf(true),
			j = (i+1)%3,
			k = (i+2)%3;

		Omega = (I[j]-I[k])/I[i] * W0[k];

		let
			magn = sqrt(W0[i]*W0[i] + W0[j]*W0[j]),
			phase = atan2(W0[j], W0[i]);

		W = Array(3);

		W[i] = t => magn * cos(Omega*t + phase);
		W[j] = t => magn * sin(Omega*t + phase);
		W[k] = t => W0[k];

	}

	ord = I.map(x=>(x<I[0])+(x<I[1])+(x<I[2]));

	pmsign = ((ord[0]+1) % 3 == ord[1]) ? 1 : -1;

	Iord = [0,1,2].map(j=>I[ord.indexOf(j)]);

	[A, B, C] = Iord;

		if (B*c >= d*d) {

			k = sqrt((A-B)*(d*d-c*C)/(B-C)/(A*c-d*d));

			[P, Q, RR] = [
				sqrt(A*(d*d-c*C)/d/d/(A-C)) * pmsign,
				sqrt(B*(d*d-c*C)/d/d/(B-C)) * pmsign,
				sqrt(C*(c*A-d*d)/d/d/(A-C)) * pmsign
			];

			lambda = sqrt((B-C)*(c*A-d*d)/A/B/C);

			eps = F(asin(W0[ord.indexOf(1)]/(d/B*Q)), k);

			Word = [
				t => -d/A*P*cos(am(lambda*t+eps, k)),
				t => d/B*Q*sin(am(lambda*t+eps, k)),
				t => d/C*RR*Delta(am(lambda*t+eps, k), k)
			];

			W = [0,1,2].map(i=>Word[ord[i]]);

			ct = t => I[2]/d*W[2](t);
			st = t => sqrt(1-(ct(t))**2);
			cy = t => I[1]/d*W[1](t)/st(t);
			sy = t => I[0]/d*W[0](t)/st(t);

			dotphi = t => d*(sy(t)**2/I[0]+cy(t)**2/I[1]);
			dottheta = t => (st(t+1e-8)-st(t))/1e-8/ct(t);
			dotpsi = t => (sy(t+1e-8)-sy(t))/1e-8/cy(t);

			wx = t => dotphi(t)*st(t)*sy(t)+dottheta(t)*cy(t);
			wy = t => dotphi(t)*st(t)*cy(t)-dottheta(t)*sy(t);
			wz = t => dotphi(t)*ct(t) + dotpsi(t);

		} else {
			k = sqrt((B-C)*(A*c-d*d)/(A-B)/(d*d-c*C));

			[P, Q, RR] = [
				sqrt(A*(d*d-c*C)/d/d/(A-C)) * pmsign,
				sqrt(B*(c*A-d*d)/d/d/(A-B)) * pmsign,
				sqrt(C*(c*A-d*d)/d/d/(A-C)) * pmsign
			];

			lambda = sqrt((A-B)*(d*d-c*C)/A/B/C);

			eps = F(asin(W0[ord.indexOf(1)]/(d/B*Q)), k);

			Word = [
				t => -d/A*P*Delta(am(lambda*t+eps, k), k),
				t => d/B*Q*sin(am(lambda*t+eps, k)),
				t => d/C*RR*cos(am(lambda*t+eps, k))
			];

			W = [0,1,2].map(i=>Word[ord[i]]);

			ct = t => I[2]/d*W[2](t);
			st = t => sqrt(1-(ct(t))**2);
			cy = t => I[1]/d*W[1](t)/st(t);
			sy = t => I[0]/d*W[0](t)/st(t);

			dotphi = t => d*(sy(t)**2/I[0]+cy(t)**2/I[1]);
			dottheta = t => (st(t+1e-8)-st(t))/1e-8/ct(t);
			dotpsi = t => (sy(t+1e-8)-sy(t))/1e-8/cy(t);

			wx = t => dotphi(t)*st(t)*sy(t)+dottheta(t)*cy(t);
			wy = t => dotphi(t)*st(t)*cy(t)-dottheta(t)*sy(t);
			wz = t => dotphi(t)*ct(t) + dotpsi(t);

		}
	console.log("pmsign", pmsign);
	console.log("sign", B*c-d*d);
	console.log("I: ", I);
	console.log("ord: ", ord);
	console.log("ABC: ", [A,B,C]);
	console.log("d: ", d);
	console.log("c: ", c);
	console.log("k: ", k);
	console.log("PQR: ", [P,Q,RR]);
	console.log("lambda: ", lambda);
	console.log("eps: ", eps);

	return [ct,st,cy,sy,dotphi];
}

/** 9 **/

function loop() {
	gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
	models.forEach(model => model.draw());
	if (NOW_PLAYING)
		requestAnimationFrame(loop);
}

function reset() {
	var index = models.indexOf(body);
	if (index >= 0)
		models.splice(index, 1);

	body = add_ellipsoid(R[0], R[1], R[2]);

	uuu = mxn_params(R, W0);

	t = 0;
	PHI = 0;

	clearInterval(interval);
	interval = setInterval(repeat, TIMEOUT);

	NOW_PLAYING = true;
	loop();
}

var repeat = function () {
	var [ctt, stt, cyt, syt, dotphit] = uuu.map(uu => uu(t));

	PHI += dotphit * dt;

	var cpt = cos(PHI),
		spt = sin(PHI);

	body.transform = new Transform([
		cyt*cpt-ctt*spt*syt,  cyt*spt+ctt*cpt*syt, syt*stt, 0,
		-syt*cpt-ctt*spt*cyt, -syt*spt+ctt*cpt*cyt, cyt*stt, 0,
		stt*spt, -stt*cpt, ctt, 0,
		0, 0, 0, 1
	]);

	body.transform.rot(-90, ey);
	body.transform.rot(-45, ez);

	if (DRAW_LVEC){
		var L = [d*syt*stt, d*cyt*stt, d*ctt, 1];
		L = vmul(body.transform, L);
		Lvec.transform = rot2(ez, L.slice(0,3));
	}
	if (DRAW_WVEC){
		var w = [d*syt*stt/I[0], d*cyt*stt/I[1], d*ctt/I[2], 1];
		w = vmul(body.transform, w);
		Wvec.transform = rot2(ez, w.slice(0,3));
	}

	t += dt;
}

/** 10 - MAIN **/

var NOW_PLAYING = true;

var camera = new Camera(
		pers_proj(.1, 10, .2, .2),
		transl(0, 0, -1.)
	);

camera.view.rmulby(rot(-90, ex));
camera.view.rmulby(rot(-135, ez));
camera.view.rmulby(rot(45, [-1, 1, 0]));
camera.update();

var R = [0.1199375, 0.2296875, 0.4399]
var W0 = [.1, 25, .1];

var
	body,

	xaxis = add_arrow(.025, .5, .05, .1, [1, 0, 0]),
	yaxis = add_arrow(.025, .5, .05, .1, [0, 1, 0]),
	zaxis = add_arrow(.025, .5, .05, .1, [0, 0, 1]);

xaxis.transform.rot(90, ey);
yaxis.transform.rot(-90, ex);

var
	Lvec = add_arrow(.025, .5, .05, .1, [1, 1, 0]),
	Wvec = add_arrow(.025, .5, .05, .1, [1, 0, 1]);

//models.splice(models.indexOf(Lvec), 1);

var t = 0, dt = 1e-2;

var
	TIMEOUT = 20,

	DRAW_LVEC = true,
	DRAW_WVEC = true;

var interval;

var PHI = 0;

reset();
