> [Demystifying cookies and tokens – Tommi Hovi | The Security blog](https://tommihovi.com/2024/05/demystifying-cookies-and-tokens/)

# Cookies

![cookies|666][https://tommihovi.com/wp-content/uploads/2024/04/image-2.png]

HTTP是一种无状态协议。HTTP 不会记住客户端会话信息，因此客户端负责使用 cookie 存储该信息。对于某些网站，无状态行为可能是可以的，也许在用户会话期间网站不应保留任何元素或用户操作。但对于大多数交互式网站来说，cookie 是必要且必不可少的，而且我们作为网站访问者也希望网站以某种方式运行。

Cookies存储在您的浏览器上。更准确地说，它们存储在硬盘驱动器上的浏览器临时目录中。。例如，Microsoft Edge 将 Cookie 存储在以下路径中：
- `C:\Users\[username]\AppData\Local\Microsoft\Edge\User Data\Default\Network`

Cookies attributes:
- Session ID 是一个唯一的随机字符串，用于标识和匹配客户端和 Web 服务器之间的会话
- Expires 定义 Cookie 设置为过期的日期
- Domain 指定要使用 Cookie 的一个或多个域
- Path 指定要使用的 Cookie 有效的资源或路径
- 启用 HttpOnly 后，将阻止客户端 API（如 JavaScript）访问 Cookie。这减轻了跨站点脚本 （XSS） 的威胁
- 启用 Secure 后，将要求仅使用 HTTPS 发送 cookie，而不允许使用 HTTP 等未加密连接，这使得 cookie 不易受到 cookie 盗窃的影响。
- Session 会话定义 cookie 是一个临时 cookie，在浏览器关闭时过期

You can see the cookies of the site you’re browsing by right-clicking and selecting the ‘_Inspect_‘ > ‘_Application_‘ > ‘_Storage_‘ > ‘_Cookies_‘. When you select a row, you can see the values on the bottom of the page (_see the Image 5._)



# Tokens

Tokens是用于信息交换的独立且紧凑的 JSON 对象。典型的令牌是在客户端（应用程序）和其他服务（如 Web 服务器）之间使用的 JSON Web 令牌 （JWT）。详细信息取决于确切的身份验证流程。在这篇文章中，我们将使用术语 JWT（发音为“JOT”），因为它更方便，在专业文献中更广泛使用。