import reflex as rx

def navbar():
  return rx.flex(
    rx.box(
      rx.link(
        rx.image(src="/logo.png", width='60px'),
        href="/"
      )
    ),
    rx.heading("ChatPDF", color="red"),
    rx.center(
      rx.menu(
        rx.menu_button('Menu', as_='b'),
        rx.menu_list(
          rx.menu_item('Login'),
          rx.menu_item('Register'),
        )
      )
    ),
    justify_content="space-between",
    bg="white",
    padding="10px 5px",
)