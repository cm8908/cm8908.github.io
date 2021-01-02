document.addEventListener("DOMContentLoaded",

	function (event) {



		// Coptic output COPY button
		document.querySelector("#copybtn").addEventListener("click", function() {
			var tempElm = document.createElement('textarea');
			tempElm.value = document.getElementById("OutputTextBox").textContent;
			document.body.appendChild(tempElm);

			tempElm.select();
			document.execCommand("copy");
			document.body.removeChild(tempElm);
		});

		// Latin output COPY button
		document.querySelector("#copybtn_l").addEventListener("click", function() {
			var tempElm = document.createElement('textarea');
			tempElm.value = document.getElementById("input").value;
			document.body.appendChild(tempElm);

			tempElm.select();
			document.execCommand("copy");
			document.body.removeChild(tempElm);
		});

		//Below COPY button
		document.querySelector("#copybtn2").addEventListener("click", function() {
			var tempElm = document.createElement('textarea');
			tempElm.value = document.getElementById("letterputhere").value;
			document.body.appendChild(tempElm);

			tempElm.select();
			document.execCommand("copy");
			document.body.removeChild(tempElm);
		});

		//Manchu output CLEAR button
		document.querySelector("#clearbtn").addEventListener("click", function() {
			document.querySelector("#OutputTextBox").textContent = "";
		});

		//Latin output CLEAR button
		document.querySelector("#clearbtn_l").addEventListener("click", function() {
			document.querySelector("#input").value = "";
		});

		//Below CLEAR button
		document.querySelector("#clearbtn2").addEventListener("click", function() {
			document.querySelector("#letterputhere").value = "";
		});


		// Copying letter from the list
		var coptic = new Array("ⲁ", "ⲃ", "ⲅ", "ⲇ", "ⲉ", "ⲋ", "ⲍ", "ⲏ", "ⲑ", "ⲓ", "ⲕ", "ⲗ", "ⲙ", "ⲛ", "ⲝ", "ⲟ", "ⲡ", "ⲣ", "ⲥ", "ⲧ", "ⲩ", "ⲫ", "ⲭ", "ⲯ", "ⲱ", "ϣ", "ϥ", "ϧ", "ϩ", "ϫ", "ϭ", "ϯ", "ⳁ");

		for (var i = 0; i<coptic.length; i++) {

			document.querySelector("[char_attr='"+ coptic[i] +"']").addEventListener("click", function() {
				
				var chkAbove = document.getElementById("above");
				var chkBelow = document.getElementById("below");

				if (chkBelow.checked) {
					var tempElm = document.querySelector("#letterputhere");
					tempElm.value += $(this).attr("char_attr");
				}

				
				if (chkAbove.checked) {
					if (document.querySelector("#OutputTextBox").textContent === "Coptic script") {
						document.querySelector("#OutputTextBox").textContent = "";
					}
					document.querySelector("#OutputTextBox").textContent += $(this).attr("char_attr");
				}
				

			});

		}


		// Letter list hide/show button
		document.querySelector("#hidebtn").addEventListener("click", function () {
			var row = document.querySelector('div#rowforletterbox');

			if (row.style.display === "block") {
				row.style.display = "none";
			} else {
				row.style.display = "block";
			}

		});


		// Transliterating LA > MA
		function TL (event) {
			var inputText = document.getElementById("input").value;
			var outputText = new Array();

			for (var i = 0; i < inputText.length; i++) {
				if (inputText[i] == "a") {
					outputText[i] = "ⲁ";
				} else if (inputText[i] == " ") {
					outputText[i] = " ";
				} else if (inputText[i] == "\n") {
					outputText[i] = "\n";
				} else if (inputText[i] == ".") {
					outputText[i] = ".";
				} else if (inputText[i] == ",") {
					outputText[i] = ",";
				} else if (inputText[i] == "e") {
					outputText[i] = "ⲉ";
				} else if (inputText[i] == "i") {
					outputText[i] = "ⲓ";
				} else if (inputText[i] == "o") {
					outputText[i] = "ⲟ";
				} else if (inputText[i] == "u") {
					outputText[i] = "ⲩ"; 
				} else if (inputText[i] == "n") {
					outputText[i] = "ⲛ"; 
				} else if (inputText[i] == "b" || inputText[i] == "v") {
					outputText[i] = "ⲃ";
				} else if (inputText[i] == "p") {
					if (inputText[i+1] == "h") {
						outputText[i] = "ⲫ";
						outputText[i+1] = "";
						i += 1;
						}
					else if (inputText[i+1] == "s") {
						outputText[i] = "ⲯ";
						outputText[i+1] = "";
						i += 1;
						}
					else { outputText[i] = "ⲡ"; }
				} else if (inputText[i] == "j") {
					outputText[i] = "ϫ";
				} else if (inputText[i] == "f") {
					outputText[i] = "ϥ";
				} else if (inputText[i] == "r") {
					outputText[i] = "ⲣ";
				} else if (inputText[i] == "k") {
					if (inputText[i+1] == "h") {
						outputText[i] = "ⲭ";
						outputText[i+1] = "";
						i += 1;
					}
					else if (inputText[i+1] == "s") {
						outputText[i] = "ⲝ";
						outputText[i+1] = "";
						i += 1;
					}
					else { outputText[i] = "ⲕ"; }
				} else if (inputText[i] == "g") {
					outputText[i] = "ⲅ";
				} else if (inputText[i] == "h") {
					outputText[i] = "ϩ"; 
				} else if (inputText[i] == "m") {
					outputText[i] = "ⲙ";
				} else if (inputText[i] == "l") {
					outputText[i] = "ⲗ";
				} else if (inputText[i] == "t") {
					if (inputText[i+1] == "h") {
						outputText[i] = "ⲑ";
						outputText[i+1] = "";
						i += 1;
					}
					else if (inputText[i+1] == "i") {
						outputText[i] = "ϯ";
						outputText[i+1] = "";
						i += 1;
					} 
					else { outputText[i] = "ⲧ"; }
				} else if (inputText[i] == "d") {
					outputText[i] = "ⲇ"; 
				} else if (inputText[i] == "s") {
					if (inputText[i+1] == "h") {
						outputText[i] = "ϣ";
						outputText[i+1] = "";
						i += 1;
					} else if (inputText[i+1] == "t") {
						outputText[i] = "ⲋ";
						outputText[i+1] = "";
						i += 1;
					} 
					else { outputText[i] = "ⲥ"; }
				} else if (inputText[i] == "š") {
					outputText[i] = "ϣ";
				} else if (inputText[i] + inputText[i+1] + inputText[i+2] == "s͡t") {
					outputText[i] = "ⲋ";
					outputText[i+1] = "";
					outputText[i+2] = "";
					i += 2;
				} else if (inputText[i] == "ô") {
					outputText[i] = "ⲱ";
				} else if (inputText[i] == "ê") {
					outputText[i] = "ⲏ";
				} else if (inputText[i] == "x") {
					outputText[i] = "ϧ";
				} else if (inputText[i] == "z") {
					outputText[i] = "ⲍ";
				}

				if (outputText[i] == undefined) {
					outputText[i] = "";
				} 	
			}

			var result = "";

			for (i = 0; i < outputText.length; i++) {
				result += outputText[i];
			}


			document.getElementById("OutputTextBox")
					.textContent = result;

		}

		function rvTL (event) {
			var inputText = document.getElementById("OutputTextBox").textContent;
			var outputText = new Array();

			for (var i = 0; i < inputText.length; i++) {
				if (inputText[i] == "ᠠ") {
					outputText[i] = "a";
				} else if (inputText[i] == "ᡝ") {
					outputText[i] = "e";
				} else if (inputText[i] == "ᡳ") {
					outputText[i] = "i";
				} else if (inputText[i] == "ᠣ") {
					outputText[i] = "o";
				} else if (inputText[i] == "ᡠ") {
					outputText[i] = "u";
				} else if (inputText[i] == "ᡳ") {
					outputText[i] = "i";
				} else if (inputText[i] == "ᡡ") {
					outputText[i] = "ū";
				} else if (inputText[i] == "ᠨ") {
					outputText[i] = "n";
				} else if (inputText[i] == "ᠩ") {
					outputText[i] = "ng";
				} else if (inputText[i] == "ᡴ") {
					outputText[i] = "k";
				} else if (inputText[i] == "ᡤ") {
					outputText[i] = "g";
				} else if (inputText[i] == "ᡥ") {
					outputText[i] = "h";
				} else if (inputText[i] == "ᠪ") {
					outputText[i] = "b";
				} else if (inputText[i] == "ᡦ") {
					outputText[i] = "p";
				} else if (inputText[i] == "ᠰ") {
					if (inputText[i+1] == "ᡟ") {
						outputText[i] = "sy";
						i += 1;
					} else {
						outputText[i] = "s";
					}
				} else if (inputText[i] == "ᡧ") {
					outputText[i] = "š";
				} else if (inputText[i] == "ᡨ") {
					outputText[i] = "t";
				} else if (inputText[i] == "ᡩ") {
					outputText[i] = "d";
				} else if (inputText[i] == "ᠯ") {
					outputText[i] = "l";
				} else if (inputText[i] == "ᠮ") {
					outputText[i] = "m";
				} else if (inputText[i] == "ᠴ") {
					outputText[i] = "c";
				} else if (inputText[i] == "ᠵ") {
					outputText[i] = "j";
				} else if (inputText[i] == "ᠶ") {
					outputText[i] = "y";
				} else if (inputText[i] == "ᡵ") {
					outputText[i] = "r";
				} else if (inputText[i] == "ᡶ") {
					outputText[i] = "f";
				} else if (inputText[i] == "ᠸ") {
					outputText[i] = "w";
				} else if (inputText[i] == "ᠺ") {
					outputText[i] = "k\'";
				} else if (inputText[i] == "ᡬ") {
					outputText[i] = "g\'";
				} else if (inputText[i] == "ᡭ") {
					outputText[i] = "h\'";
				} else if (inputText[i] == "ᡮ") {
					if (inputText[i+1] == "ᡟ") {
						outputText[i] = "ts";
						i += 1;
					} else {
						outputText[i] = "ts\'";
					}
				} else if (inputText[i] == "ᡯ") {
					outputText[i] = "dz";
				} else if (inputText[i] == "ᡰ") {
					outputText[i] = "ž";
				} else if (inputText[i] == "ᡱ") {
					if (intputText[i+1] == "ᡳ") {
						outputText[i] = "c\'y";
						i += 1;
					} else {
						outputText[i] = "c\'";
					}
				} else if (inputText[i] == "ᡷ") {
					if (intputText[i+1] == "ᡳ") {
						outputText[i] = "jy";
						i += 1;
					} else {
						outputText[i] = "zh";
					}
					outputText[i] = "zh";
				} else if (inputText[i] == " ") {
					outputText[i] = " ";
				} else if (inputText[i] == "\n") {
					outputText[i] = "\n";
				} else if (inputText[i] == "᠉") {
					outputText[i] = ".";
				} else if (inputText[i] == "᠈") {
					outputText[i] = ",";
				} 

				if (outputText[i] == undefined) {
					outputText[i] = "";
				} 	
			}

			var result = "";

			for (i = 0; i < outputText.length; i++) {
				result += outputText[i];
			}


			document.getElementById("input")
					.value = result;

		}

		document.querySelector("button#tl")
				.addEventListener("click", TL);

		document.querySelector("button#rvtl")
				.addEventListener("click", rvTL);

		// document.querySelector("button#hidebtn")
		// 		.addEventListener("click", hideshow());


		window.addEventListener("keyup", function(event) {
			if (event.keyCode == 13) {
				document.querySelector("button#tl").click();
			}
		});

	}
);