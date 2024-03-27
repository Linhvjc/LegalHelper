import { TypeAnimation } from "react-type-animation";

const TypingAnimAdmin = () => {
  return (
    <TypeAnimation
      sequence={[
        // Same substring at the start will only be typed once, initially
        "Welcome to admin page",
        1000,
        "Built With Your Model 🤖",
        2000,
        "Your Own Customized 💻",
        1500,
      ]}
      speed={50}
      style={{
        fontSize: "60px",
        color: "white",
        display: "inline-block",
        textShadow: "1px 1px 20px #000",
      }}
      repeat={Infinity}
    />
  );
};

export default TypingAnimAdmin;
