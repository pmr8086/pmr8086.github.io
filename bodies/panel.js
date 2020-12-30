document.getElementById("btn_omega").addEventListener("change", function () {
	var index;
	if ((index = models.indexOf(Wvec)) >= 0 ) {
		DRAW_WVEC = false;
		models.splice(models.indexOf(Wvec), 1);
		requestAnimationFrame(loop);
	}
	else {
		DRAW_WVEC = true;
		models.push(Wvec);
		requestAnimationFrame(loop);
	}
});
document.getElementById("btn_L").addEventListener("change", function () {
	var index;
	if ((index = models.indexOf(Lvec)) >= 0 ) {
		DRAW_LVEC = false;
		models.splice(models.indexOf(Lvec), 1);
		requestAnimationFrame(loop);
	}
	else {
		DRAW_LVEC = true;
		models.push(Lvec);
		requestAnimationFrame(loop);
	}
});
document.getElementById("btn_pause").addEventListener("click", function () {
	if (interval) {
		NOW_PLAYING = false;
		clearInterval(interval);
		interval = false;
	} else {
		NOW_PLAYING = true;
		loop();
		interval = setInterval(repeat, TIMEOUT);
	}
});
document.getElementById("btn_restart").addEventListener("click", function() {
	R[0] = document.getElementById("slider_r1").value;
	R[1] = document.getElementById("slider_r2").value;
	R[2] = document.getElementById("slider_r3").value;
	W0[0] = document.getElementById("slider_w01").value;
	W0[1] = document.getElementById("slider_w02").value;
	W0[2] = document.getElementById("slider_w03").value;
	reset();
});
