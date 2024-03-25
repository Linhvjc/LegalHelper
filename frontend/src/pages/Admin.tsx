import { Box, useMediaQuery, useTheme } from "@mui/material";
import React from "react";
import TypingAnimAdmin from "../components/typer/TypingAnimAdmin";
import StickyHeadTableParam from "../components/table/StickyHeadTableParam";
import CollapsibleTable from "../components/table/CollapsibleTable";


const Home = () => {
  const theme = useTheme();
  const isBelowMd = useMediaQuery(theme.breakpoints.down("md"));
  return (
    <Box width={"100%"} height={"100%"}>
      <Box
        sx={{
          display: "flex",
          width: "100%",
          flexDirection: "column",
          alignItems: "center",
          mx: "auto",
          mt: 3,
        }}
      >
        <Box>
          <TypingAnimAdmin />
        </Box>
        <Box
          sx={{
            width: "80%",
            display: "flex",
            flexDirection: { md: "row", xs: "column", sm: "column" },
            gap: 5,
            my: 10,
          }}
        >
          <StickyHeadTableParam />
        </Box>
        <Box
          sx={{
            width: "80%",
            display: "flex",
            flexDirection: { md: "row", xs: "column", sm: "column" },
            gap: 5,
            my: 10,
          }}
        >
          <StickyHeadTableParam />
        </Box>

        <Box
          sx={{
            width: "80%",
            display: "flex",
            flexDirection: { md: "row", xs: "column", sm: "column" },
            gap: 5,
            my: 10,
          }}
        >
          <CollapsibleTable />
        </Box>
      </Box>
      {/* <Footer /> */}
    </Box>
  );
};

export default Home;
