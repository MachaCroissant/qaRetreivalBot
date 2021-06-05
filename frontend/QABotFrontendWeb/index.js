document.addEventListener("DOMContentLoaded", () => {
  const inputField = document.getElementById("input");
  inputField.addEventListener("keydown", (e) => {
    if (e.code === "Enter"){
      let input = inputField.value;
      inputField.value = "";
      output(input);
    }
  });
});

function createRandomId() {
  return (Math.random()*10000000).toString(16).substr(0,4)+'-'+(new Date()).getTime()+'-'+Math.random().toString().substr(2,5);
}

function output(query) {
  let product;
  let request_id = createRandomId();
  console.log(request_id)
  let queryObj = {"request_id": request_id, "query": query};
  console.log(queryObj)
  let url = "http://127.0.0.1:8080/test?ques=" + query;
  console.log(url)
  fetch(url)
  .then(function (response) {
    response.json().then(function(data){
      product = data.answer;
      addChat(query, product);
    });
  }).catch(function (error) {
    console.log('Request failed', error);})

}

function addChat(input, product){
  const messageContainer = document.getElementById("messages");
  
  let userDiv = document.createElement("div");
  userDiv.id = "user";
  userDiv.className="user response";
  userDiv.innerHTML=`<img src="usericon.png" class="avatar"><span>${input}</span>`;
  messageContainer.appendChild(userDiv);

  let botDiv = document.createElement("div");
  botDiv.id = "bot";
  botDiv.className = "bot response";
  let botImg = document.createElement("img");
  botImg.src = "boticon.png";
  botImg.className = "avatar";
  let botLoadingText = document.createElement("div");
  botLoadingText.className="dot-elastic"
  
  botDiv.append(botLoadingText);
  botDiv.append(botImg);
  
  messageContainer.appendChild(botDiv);

  messageContainer.scrollTop = messageContainer.scrollHeight - messageContainer.clientHeight

  setTimeout(() => {
    botLoadingText.removeAttribute("class")
    botLoadingText.innerText = `${product}`;
  }, 2000)

}