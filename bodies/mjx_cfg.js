window.MathJax = {
	extensions: ["tex2jax.js"],
	jax: ["input/TeX", "output/HTML-CSS"],
	tex2jax: {
		inlineMath: [['$', '$']]
	},
	TeX: {
		extensions: ["AMSsymbols.js"],
		Macros: {
			epsilon: ["\\varepsilon"],
			cn: ["\\mathop{\\rm cn}\\nolimits"],
			sn: ["\\mathop{\\rm sn}\\nolimits"],
			dn: ["\\mathop{\\rm dn}\\nolimits"],
			am: ["\\mathop{\\rm am}\\nolimits"],
			gd: ["\\mathop{\\rm gd}\\nolimits"],
			sech: ["\\mathop{\\rm sech}\\nolimits"]
		}
	}
}